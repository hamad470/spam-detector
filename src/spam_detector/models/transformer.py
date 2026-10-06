"""Fine-tuned DistilRoBERTa.

    python -m spam_detector.models.transformer --augment

A plain PyTorch loop rather than the HF Trainer: the dataset is small, and this
keeps the moving parts (class weights, early stopping, temperature) visible.
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
from ..paths import ARTIFACTS, LABELS, PROCESSED
from ..text import prepare

BASE = "distilroberta-base"
MAX_LEN = 96  # 99% of messages fit in 80 tokens


def augment(texts, labels, seed=0):
    """Add an obfuscated copy of every spam/smishing message and a fifth of ham.

    Ham gets copies too, otherwise the model learns "weird spelling means spam".
    """
    rng = random.Random(seed)
    extra = [
        (mixed(t, rng.uniform(0.15, 0.5), rng), y)
        for t, y in zip(texts, labels, strict=True)
        if y != "ham" or rng.random() < 0.2
    ]
    return list(texts) + [t for t, _ in extra], list(labels) + [y for _, y in extra]


class DistilRobertaClassifier:
    name = "distilroberta"

    def __init__(self, path: str | Path | None = None, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.temperature = 1.0
        if path:
            path = Path(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
            meta = path / "calibration.json"
            if meta.exists():
                self.temperature = json.loads(meta.read_text())["temperature"]

    def _batches(self, texts, batch_size):
        for i in range(0, len(texts), batch_size):
            chunk = [prepare(t) for t in texts[i : i + batch_size]]
            yield self.tokenizer(
                chunk, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
            )

    @torch.no_grad()
    def logits(self, texts, batch_size=64) -> np.ndarray:
        self.model.eval()
        out = [self.model(**b).logits for b in self._batches(list(texts), batch_size)]
        return torch.cat(out).numpy()

    def predict_proba(self, texts) -> np.ndarray:
        z = self.logits(texts) / self.temperature
        z = z - z.max(1, keepdims=True)
        return np.exp(z) / np.exp(z).sum(1, keepdims=True)

    def fit(
        self, train_texts, train_labels, val_texts, val_labels, epochs=4, lr=4e-5, batch_size=16, seed=42
    ):
        torch.manual_seed(seed)
        random.seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BASE,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={name: i for i, name in enumerate(LABELS)},
        )
        y = torch.tensor([LABELS.index(label) for label in train_labels])
        # Square-root inverse frequency: helps the two small classes without wrecking ham precision
        counts = torch.bincount(y, minlength=len(LABELS)).float()
        weights = (counts.sum() / counts).sqrt()
        weights /= weights.mean()

        steps = epochs * math.ceil(len(train_texts) / batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

        best, best_state, history = -1.0, None, []
        for epoch in range(epochs):
            self.model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(train_texts))
            texts = [train_texts[i] for i in order]
            started, running = time.time(), 0.0
            for step, batch in enumerate(self._batches(texts, batch_size)):
                yb = y[order[step * batch_size : (step + 1) * batch_size]]
                loss = F.cross_entropy(self.model(**batch).logits, yb, weight=weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                running += loss.item()
            val_pred = self.logits(val_texts).argmax(1)
            val_f1 = f1_score([LABELS.index(v) for v in val_labels], val_pred, average="macro")
            history.append({"epoch": epoch + 1, "train_loss": running / (step + 1), "val_macro_f1": val_f1})
            print(f"epoch {epoch + 1}: loss {running / (step + 1):.4f}  val macro-F1 {val_f1:.4f}  "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)  # fmt: skip
            if val_f1 > best:
                best = val_f1
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        self.history = history
        self.temperature = self._fit_temperature(val_texts, val_labels)
        return self

    def _fit_temperature(self, texts, labels) -> float:
        """Single scalar T minimising validation NLL (Guo et al., 2017)."""
        z = torch.tensor(self.logits(texts))
        y = torch.tensor([LABELS.index(label) for label in labels])
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        return float(log_t.exp())

    def save(self, path: str | Path):
        path = Path(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "calibration.json").write_text(json.dumps({"temperature": self.temperature}))
        (path / "history.json").write_text(json.dumps(self.history, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--augment", action="store_true", help="add obfuscated copies of training messages")
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    df = pd.read_parquet(PROCESSED / "corpus.parquet")
    df = df[df.label.isin(LABELS)]
    train, val = df[df.split == "train"], df[df.split == "val"]
    texts, labels = list(train.text), list(train.label)
    if args.augment:
        texts, labels = augment(texts, labels)
    print(f"training on {len(texts)} messages ({'with' if args.augment else 'without'} augmentation)")

    clf = DistilRobertaClassifier().fit(texts, labels, list(val.text), list(val.label), epochs=args.epochs)
    out = ARTIFACTS / ("distilroberta_aug" if args.augment else "distilroberta")
    clf.save(out)
    print(f"saved to {out} (temperature {clf.temperature:.3f})")


if __name__ == "__main__":
    main()
