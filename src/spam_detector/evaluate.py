"""Run every experiment in the report and write reports/results.json.

    python -m spam_detector.evaluate

Classical models are trained here (seconds); transformers are loaded from
artifacts/ if they've been trained. Everything is scored on the campaign-held-out
test split unless stated otherwise.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .attacks import attack_all
from .datasets import load_enron, load_mendeley, load_uci
from .decision import decide
from .models.classical import NBOriginal, TfidfLR
from .models.embeddings import EmbeddingLR
from .paths import ARTIFACTS, LABELS, PROCESSED, REPORTS
from .text import dedup_key

SEED = 42
ATTACKS = ["leetspeak", "homoglyph", "spacing", "typo", "mixed"]
RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]


def malicious_score(proba: np.ndarray) -> np.ndarray:
    """P(spam) + P(smishing): the score a 'block or allow' filter would threshold.

    Written as 1 - P(ham) so it also works for the binary models in the leakage study.
    """
    return 1.0 - proba[:, 0]


def expected_calibration_error(y_true, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def bootstrap_ci(y_true, y_pred, metric, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = [
        metric(y_true[s], y_pred[s]) for s in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))
    ]
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def latency_ms(
    model, text="URGENT! Your parcel is held. Pay the £1.99 fee at royalmail-redelivery.com", runs=30
):
    model.predict_proba([text])  # warm-up
    start = time.perf_counter()
    for _ in range(runs):
        model.predict_proba([text])
    return round((time.perf_counter() - start) / runs * 1000, 2)


def load_models(train: pd.DataFrame) -> dict:
    models = {}
    for m in [NBOriginal(), TfidfLR(), EmbeddingLR()]:
        started = time.time()
        models[m.name] = m.fit(list(train.text), list(train.label))
        print(f"trained {m.name} in {time.time() - started:.1f}s", flush=True)
        joblib.dump(m, ARTIFACTS / f"{m.name}.joblib", compress=3)
    for name in ["distilroberta", "distilroberta_aug"]:
        if (ARTIFACTS / name / "config.json").exists():
            from .models.transformer import DistilRobertaClassifier

            models[name] = DistilRobertaClassifier(ARTIFACTS / name)
            models[name].name = name
    return models


def size_mb(name: str) -> float:
    path = ARTIFACTS / name
    files = [path / "model.safetensors"] if path.is_dir() else [path.with_suffix(".joblib")]
    return round(sum(f.stat().st_size for f in files if f.exists()) / 1e6, 1)


def leakage_experiment() -> dict:
    """How much do naive merges and random splits flatter a model?

    Same model (tfidf_lr), same binary task, three protocols. Only the last one
    resembles deployment, where next month's campaigns are new.
    """
    uci, men = load_uci(), load_mendeley()
    naive = pd.concat([uci, men], ignore_index=True)
    naive["y"] = (naive.label != "ham").astype(int)
    exact = naive.assign(key=naive.text.map(dedup_key)).drop_duplicates("key")

    def run(df, seed):
        tr, te = train_test_split(df, test_size=0.15, stratify=df.y, random_state=seed)
        m = TfidfLR().fit(list(tr.text), np.where(tr.y, "spam", "ham"))
        return f1_score(te.y, malicious_score(m.predict_proba(list(te.text))) >= 0.5)

    out = {}
    for name, df in [("naive_merge_random_split", naive), ("exact_dedup_random_split", exact)]:
        scores = [run(df, s) for s in range(3)]
        out[name] = {"f1_mean": round(float(np.mean(scores)), 4), "f1_std": round(float(np.std(scores)), 4)}
        print(name, out[name], flush=True)

    corpus = pd.read_parquet(PROCESSED / "corpus.parquet")
    corpus = corpus[corpus.label.isin(LABELS)]
    tr, te = corpus[corpus.split != "test"], corpus[corpus.split == "test"]
    m = TfidfLR().fit(list(tr.text), list(tr.label))
    f1 = f1_score(te.label != "ham", malicious_score(m.predict_proba(list(te.text))) >= 0.5)
    out["near_dup_grouped_split"] = {"f1_mean": round(float(f1), 4), "f1_std": 0.0}
    # How many test messages under the naive protocol have an exact copy in its training set
    tr_n, te_n = train_test_split(naive, test_size=0.15, stratify=naive.y, random_state=0)
    seen = set(tr_n.text.map(dedup_key))
    out["naive_test_messages_with_exact_copy_in_train"] = round(
        float(te_n.text.map(dedup_key).isin(seen).mean()), 3
    )
    return out


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED / "corpus.parquet")
    labelled = df[df.label.isin(LABELS)]
    train = labelled[labelled.split == "train"]
    test = labelled[labelled.split == "test"].reset_index(drop=True)
    unlabelled_spam = df[(df.label == "spam_unlabelled") & (df.split == "test")]
    enron = load_enron()

    results = {"test_size": test.label.value_counts().to_dict(), "models": {}}
    y = np.array([LABELS.index(label) for label in test.label])
    y_bin = (y > 0).astype(int)
    new_only = (test.source == "mendeley_only").to_numpy()
    preds = {"label": test.label, "source": test.source, "text": test.text}

    models = load_models(train)
    for name, model in models.items():
        print(f"evaluating {name}", flush=True)
        proba = model.predict_proba(list(test.text))
        pred, score = decide(proba), malicious_score(proba)
        pred_bin = (score >= 0.5).astype(int)
        preds[f"{name}__malicious"] = score
        for i, label in enumerate(LABELS):
            preds[f"{name}__p_{label}"] = proba[:, i]

        r = {
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "macro_f1_ci95": bootstrap_ci(y, pred, lambda a, b: f1_score(a, b, average="macro")),
            "per_class": {
                k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score")}
                for k, v in classification_report(y, pred, target_names=LABELS, output_dict=True).items()
                if k in LABELS
            },
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
            "binary": {
                "precision": round(precision_score(y_bin, pred_bin), 4),
                "recall": round(recall_score(y_bin, pred_bin), 4),
                "f1": round(f1_score(y_bin, pred_bin), 4),
                "f1_ci95": bootstrap_ci(y_bin, pred_bin, f1_score),
                "pr_auc": round(average_precision_score(y_bin, score), 4),
                "roc_auc": round(roc_auc_score(y_bin, score), 4),
                "ham_false_positive_rate": round(float(pred_bin[y_bin == 0].mean()), 4),
                "ece": round(expected_calibration_error(y_bin, score), 4),
            },
            "new_2022_messages_macro_f1": round(f1_score(y[new_only], pred[new_only], average="macro"), 4),
            "unlabelled_uci_spam_detection_rate": round(
                float((malicious_score(model.predict_proba(list(unlabelled_spam.text))) >= 0.5).mean()), 4
            ),
            "latency_ms": latency_ms(model),
            "size_mb": size_mb(name),
        }

        enron_score = malicious_score(model.predict_proba(list(enron.text)))
        enron_y = (enron.label == "spam").to_numpy().astype(int)
        r["enron_email"] = {
            "roc_auc": round(roc_auc_score(enron_y, enron_score), 4),
            "f1": round(f1_score(enron_y, enron_score >= 0.5), 4),
        }

        malicious = test[test.label != "ham"].text.tolist()
        ham = test[test.label == "ham"].text.tolist()
        r["robustness"] = {}
        for attack in ATTACKS:
            r["robustness"][attack] = {}
            for rate in RATES:
                adv = attack_all(malicious, attack, rate, seed=1)
                recall = float((malicious_score(model.predict_proba(adv)) >= 0.5).mean())
                r["robustness"][attack][str(rate)] = round(recall, 4)
        # Obfuscated ham shouldn't suddenly look like spam either
        r["robustness_ham_fpr_mixed_0.3"] = round(
            float(
                (malicious_score(model.predict_proba(attack_all(ham, "mixed", 0.3, seed=1))) >= 0.5).mean()
            ),
            4,
        )
        results["models"][name] = r
        print(json.dumps({k: r[k] for k in ("macro_f1", "binary", "latency_ms")}), flush=True)

    # Ablation: same tfidf_lr without the normaliser, to isolate what normalisation buys under attack
    raw = TfidfLR(normalise_input=False).fit(list(train.text), list(train.label))
    malicious = test[test.label != "ham"].text.tolist()
    results["ablation_tfidf_lr_without_normaliser"] = {
        attack: {
            str(rate): round(
                float(
                    (
                        malicious_score(raw.predict_proba(attack_all(malicious, attack, rate, seed=1))) >= 0.5
                    ).mean()
                ),
                4,
            )
            for rate in RATES
        }
        for attack in ATTACKS
    }

    results["leakage"] = leakage_experiment()
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(preds).to_parquet(REPORTS / "test_predictions.parquet", index=False)
    print("wrote reports/results.json")


if __name__ == "__main__":
    main()
