"""Fine-tune DistilRoBERTa on SMS and email.

    python -m spam_detector.models.transformer --seed 0                    # main protocol
    python -m spam_detector.models.transformer --domains sms --seed 0      # one channel only
    python -m spam_detector.models.transformer --final --save-model        # deployed model

Written to run unchanged on a laptop CPU or a Colab GPU (notebooks/train_on_colab.ipynb).
A plain PyTorch loop rather than the HF Trainer: class weights, channel-balanced sampling,
early stopping and the temperature are all visible here.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ..attacks import mixed
from ..decision import decide
from ..paths import ARTIFACTS, LABELS, PROCESSED
from ..text import prepare

BASE = "distilroberta-base"
MAX_LEN = {"sms": 96, "email": 256}  # 99% of SMS fit in 80 tokens; emails are cut after ~200 words


def augment(texts, domains, labels, seed=0):
    """Add an obfuscated copy of every spam/phishing message and of a fifth of legit ones.

    Legit messages get copies too, otherwise the model learns "odd spelling means scam".
    """
    rng = random.Random(seed)
    extra = [
        (mixed(t, rng.uniform(0.15, 0.5), rng), d, y)
        for t, d, y in zip(texts, domains, labels, strict=True)
        if y != "legit" or rng.random() < 0.2
    ]
    return (
        list(texts) + [e[0] for e in extra],
        list(domains) + [e[1] for e in extra],
        list(labels) + [e[2] for e in extra],
    )


def macro_f1_by_domain(y_true, y_pred, domains) -> dict:
    out = {}
    for d in sorted(set(domains)):
        m = np.asarray(domains) == d
        out[d] = f1_score(np.asarray(y_true)[m], np.asarray(y_pred)[m], average="macro", labels=[0, 1, 2])
    out["mean"] = float(np.mean(list(out.values())))
    return out


class DistilRobertaClassifier:
    name = "distilroberta"

    def __init__(self, path: str | Path | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = 1.0
        if path:
            path = Path(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
            meta = path / "calibration.json"
            if meta.exists():
                self.temperature = json.loads(meta.read_text())["temperature"]

    # -- batching ------------------------------------------------------------

    def _encode(self, prepared: list[str], domains: list[str]):
        ids = []
        for text, d in zip(prepared, domains, strict=True):
            ids.append(self.tokenizer(text, truncation=True, max_length=MAX_LEN[d])["input_ids"])
        return ids

    def _collate(self, ids_list):
        width = max(len(x) for x in ids_list)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(ids_list), width), pad, dtype=torch.long)
        mask = torch.zeros((len(ids_list), width), dtype=torch.long)
        for i, x in enumerate(ids_list):
            ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return ids.to(self.device), mask.to(self.device)

    @staticmethod
    def _length_batches(lengths, batch_size, rng=None):
        """Batches of similar length, so short SMS aren't padded to email length."""
        order = np.argsort(lengths, kind="stable")
        batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        if rng is not None:
            rng.shuffle(batches)
        return batches

    @torch.no_grad()
    def logits(self, texts, domains, batch_size=64) -> np.ndarray:
        self.model.eval()
        ids = self._encode([prepare(t, d) for t, d in zip(texts, domains, strict=True)], list(domains))
        out = np.zeros((len(ids), len(LABELS)), dtype=np.float32)
        for batch in self._length_batches([len(x) for x in ids], batch_size):
            input_ids, mask = self._collate([ids[i] for i in batch])
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                out[batch] = self.model(input_ids=input_ids, attention_mask=mask).logits.float().cpu().numpy()
        return out

    def predict_proba(self, texts, domains) -> np.ndarray:
        z = self.logits(texts, domains) / self.temperature
        z = z - z.max(1, keepdims=True)
        return np.exp(z) / np.exp(z).sum(1, keepdims=True)

    # -- training ------------------------------------------------------------

    def fit(
        self, train: pd.DataFrame, val: pd.DataFrame, epochs=3, lr=4e-5, batch_size=32, seed=0, log=print
    ):
        torch.manual_seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BASE,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={name: i for i, name in enumerate(LABELS)},
        ).to(self.device)

        texts, domains, labels = augment(
            train.text.tolist(), train.domain.tolist(), train.label.tolist(), seed
        )
        ids = self._encode([prepare(t, d) for t, d in zip(texts, domains, strict=True)], domains)
        y = torch.tensor([LABELS.index(label) for label in labels])
        dom = np.asarray(domains)

        # Square-root inverse frequency for classes; channels resampled the same way so the
        # ~4x larger email set doesn't drown out SMS
        counts = torch.bincount(y, minlength=len(LABELS)).float()
        class_w = (counts.sum() / counts).sqrt()
        class_w = (class_w / class_w.mean()).to(self.device)
        dom_n = {d: (dom == d).sum() for d in set(domains)}
        sample_p = np.array([1 / math.sqrt(dom_n[d]) for d in domains])
        sample_p /= sample_p.sum()

        steps_per_epoch = math.ceil(len(ids) / batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(
            opt, int(0.06 * epochs * steps_per_epoch), epochs * steps_per_epoch
        )
        scaler = torch.amp.GradScaler(enabled=self.device == "cuda")

        best, best_state, self.history = -1.0, None, []
        y_val = [LABELS.index(v) for v in val.label]
        for epoch in range(epochs):
            self.model.train()
            started, running = time.time(), 0.0
            drawn = rng.choice(len(ids), size=len(ids), replace=True, p=sample_p)
            batches = self._length_batches([len(ids[i]) for i in drawn], batch_size, rng)
            for batch in batches:
                rows = drawn[batch]
                input_ids, mask = self._collate([ids[i] for i in rows])
                with torch.autocast(self.device, enabled=self.device == "cuda"):
                    logits = self.model(input_ids=input_ids, attention_mask=mask).logits
                loss = F.cross_entropy(logits.float(), y[rows].to(self.device), weight=class_w)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
                running += loss.item()
            val_prob = torch.softmax(torch.from_numpy(self.logits(val.text, val.domain)), dim=1).numpy()
            f1 = macro_f1_by_domain(y_val, decide(val_prob), val.domain.tolist())
            loss_avg = running / len(batches)
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": loss_avg,
                    **{f"val_f1_{k}": v for k, v in f1.items()},
                }
            )
            scores = "  ".join(f"{k} {v:.4f}" for k, v in f1.items())
            minutes = (time.time() - started) / 60
            log(f"epoch {epoch + 1}: loss {loss_avg:.4f}  val macro-F1 {scores}  ({minutes:.1f} min)")
            # Select on the mean over channels so the bigger email set can't hide an SMS regression
            if f1["mean"] > best:
                best = f1["mean"]
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        self.temperature = self._fit_temperature(val)
        return self

    def _fit_temperature(self, val) -> float:
        """Single scalar T minimising validation NLL (Guo et al., 2017)."""
        z = torch.tensor(self.logits(val.text, val.domain))
        y = torch.tensor([LABELS.index(label) for label in val.label])
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        return float(log_t.detach().exp())

    def save(self, path: str | Path):
        path = Path(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "calibration.json").write_text(json.dumps({"temperature": self.temperature}))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--domains", default="sms,email", help="channels to train on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--final", action="store_true", help="train on train+test+future for deployment")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="subsample for smoke tests")
    p.add_argument("--corpus", default=PROCESSED / "corpus.parquet", type=Path)
    p.add_argument("--out", default=ARTIFACTS / "runs", type=Path)
    args = p.parse_args(argv)

    df = pd.read_parquet(
        args.corpus, columns=["text", "label", "domain", "source", "year", "split", "future_novel"]
    )
    labelled = df[df.label.isin(LABELS)]
    domains = args.domains.split(",")
    train_splits = ["train", "test", "future"] if args.final else ["train"]
    train = labelled[labelled.split.isin(train_splits) & labelled.domain.isin(domains)]
    val = labelled[labelled.split == "val"]  # both channels, so single-channel runs are still selected fairly
    if args.limit:
        train = train.groupby(["domain", "label"]).head(args.limit // 6)
        val = val.groupby(["domain", "label"]).head(args.limit // 12)

    name = ("final" if args.final else "-".join(domains)) + f"_seed{args.seed}"
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / "train.log").open("w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log(f"{name}: {len(train)} training messages {train.groupby(['domain', 'label']).size().to_dict()}")
    clf = DistilRobertaClassifier().fit(
        train, val, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, log=log
    )
    log(f"temperature {clf.temperature:.3f}")

    # Score every evaluation split here, so the slow part never has to run on a CPU
    evaluate_on = df[df.split.isin(["val", "test", "future"])]
    if args.limit:
        evaluate_on = evaluate_on.groupby(["split", "domain", "label"]).head(20)
    probs = clf.predict_proba(evaluate_on.text, evaluate_on.domain)
    pred = evaluate_on[["split", "domain", "label", "source", "year", "future_novel"]].copy()
    pred[[f"p_{label}" for label in LABELS]] = probs
    pred.to_parquet(out / "predictions.parquet")
    (out / "history.json").write_text(json.dumps(clf.history, indent=2))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "domains": domains,
                "seed": args.seed,
                "final": args.final,
                "temperature": clf.temperature,
                "epochs": args.epochs,
                "train_size": len(train),
            },
            indent=2,
        )  # fmt: skip
    )
    if args.save_model:
        clf.save(out / "model")
    log(f"saved {out}")


if __name__ == "__main__":
    main()
