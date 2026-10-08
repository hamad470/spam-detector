"""Run every experiment in the report and write reports/results.json.

    python -m spam_detector.evaluate

Baselines are trained here (minutes on a CPU). Transformer results are read from
artifacts/runs/*/predictions.parquet, written by the Colab notebook.

Test sets:
  test          held-out campaigns, SMS and email, 2003-2022
  future        Nazario phishing from 2023-2025, never trained on (except by the final model)
  no-hook       test scams with no link, phone number, email address or money amount
"""

import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from .attacks import attack_all
from .datasets import load_mendeley, load_uci
from .decision import decide
from .models.classical import NBOriginal, TfidfLR
from .paths import ARTIFACTS, DOMAINS, LABELS, MODELS, PROCESSED, REPORTS
from .text import dedup_key, prepare

SEED = 42
ATTACKS = ["leetspeak", "homoglyph", "spacing", "typo", "mixed"]
RATES = [0.0, 0.1, 0.3, 0.5, 0.7]
HOOKS = re.compile(r"\bxx(?:url|phone|money|email)\b")
RUNS = ARTIFACTS / "runs"
MAIN_SEEDS = ["sms-email_seed0", "sms-email_seed1", "sms-email_seed2"]


def scam_score(proba: np.ndarray) -> np.ndarray:
    """P(spam) + P(phishing), written as 1 - P(legit) so it also works for binary models."""
    return 1.0 - proba[:, 0]


def expected_calibration_error(y_true, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(
        sum(
            (idx == b).mean() * abs(y_true[idx == b].mean() - p[idx == b].mean())
            for b in range(bins)
            if (idx == b).any()
        )
    )


def bootstrap_ci(y_true, y_pred, metric, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = [
        metric(y_true[s], y_pred[s]) for s in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))
    ]
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def channel_metrics(frame: pd.DataFrame, proba: np.ndarray) -> dict:
    """Everything reported for one model on one channel's test set."""
    y = np.array([LABELS.index(label) for label in frame.label])
    pred = decide(proba)
    # Average F1 over the classes this test set actually has (ExAIS has no phishing class)
    present = sorted(set(y.tolist()))
    is_scam, flagged, score = y > 0, pred > 0, scam_score(proba)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "n": len(frame),
        "macro_f1": round(f1_score(y, pred, average="macro", labels=present), 4),
        "macro_f1_ci95": bootstrap_ci(y, pred, lambda a, b: f1_score(a, b, average="macro", labels=present)),
        "per_class": {
            label: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4)}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "scam_f1": round(f1_score(is_scam, flagged), 4),
        "scam_precision": round(precision_score(is_scam, flagged, zero_division=0), 4),
        "scam_recall": round(recall_score(is_scam, flagged), 4),
        "legit_false_positive_rate": round(float(flagged[~is_scam].mean()), 4),
        "pr_auc": round(average_precision_score(is_scam, score), 4),
        "ece": round(expected_calibration_error(is_scam.astype(float), score), 4),
    }


def scam_recall(proba) -> float:
    return round(float((decide(proba) > 0).mean()), 4)


def phishing_recall(proba) -> float:
    return round(float((decide(proba) == 2).mean()), 4)


def evaluate_predictions(df: pd.DataFrame, proba: np.ndarray) -> dict:
    """df: rows of the val/test/future splits, aligned with proba."""
    out = {}
    lab = df.label.isin(LABELS).to_numpy()
    for d in DOMAINS:
        m = ((df.split == "test") & (df.domain == d)).to_numpy() & lab
        out[d] = channel_metrics(df[m], proba[m])
    # ExAIS is never trained on (its labels follow a different definition of spam); it is
    # reported as an external test. The SMS test split is UCI/Mendeley only.
    out["sms_uci_mendeley"] = out["sms"]
    ext = (df.split == "external").to_numpy() & lab
    out["sms_exais"] = channel_metrics(df[ext], proba[ext])
    fut = (df.split == "future").to_numpy()
    novel = fut & df.future_novel.to_numpy()
    out["future"] = {
        "n": int(fut.sum()),
        "phishing_recall": phishing_recall(proba[fut]),
        "scam_recall": scam_recall(proba[fut]),
        "novel_n": int(novel.sum()),
        "novel_phishing_recall": phishing_recall(proba[novel]),
        "by_year": {
            int(y): scam_recall(proba[fut & (df.year == y).to_numpy()]) for y in sorted(df[fut].year.unique())
        },
    }
    # Older phishing in the test split, by year, to put the future numbers in context
    old = ((df.split == "test") & (df.source == "nazario")).to_numpy()
    out["test_phishing_by_era"] = {
        "2005-2007": scam_recall(proba[old & (df.year <= 2007).to_numpy()]),
        "2015-2022": scam_recall(proba[old & (df.year >= 2015).to_numpy()]),
    }
    test = (df.split == "test").to_numpy() & lab
    no_hook = test & ~df.prepared.str.contains(HOOKS).to_numpy()
    out["no_hook"] = {}
    for d in DOMAINS:
        dm = no_hook & (df.domain == d).to_numpy()
        scams = dm & (df.label != "legit").to_numpy()
        legit = dm & (df.label == "legit").to_numpy()
        out["no_hook"][d] = {
            "scams": int(scams.sum()),
            "scam_recall": scam_recall(proba[scams]),
            "legit": int(legit.sum()),
            "legit_false_positive_rate": round(float((decide(proba[legit]) > 0).mean()), 4),
            "scam_recall_with_hooks": scam_recall(
                proba[test & ~no_hook & (df.domain == d).to_numpy() & (df.label != "legit").to_numpy()]
            ),
        }
    unl = ((df.split == "test") & (df.label == "spam_unlabelled")).to_numpy()
    out["uci_spam_without_subtype_detected"] = scam_recall(proba[unl])
    return out


def latency_ms(
    model, text="URGENT! Your parcel is held. Pay the £1.99 fee at royalmail-redelivery.com", runs=30
):
    model.predict_proba([text], ["sms"])
    start = time.perf_counter()
    for _ in range(runs):
        model.predict_proba([text], ["sms"])
    return round((time.perf_counter() - start) / runs * 1000, 2)


def robustness(model, val: pd.DataFrame) -> dict:
    """Share of validation scams still flagged as the share of disguised words grows.

    Validation, not test: the deployed transformer was trained on the test split, so val is the
    only clean data every model here has never been fitted on.
    """
    out = {}
    for d in DOMAINS:
        scams = val[(val.domain == d) & (val.label != "legit")]
        if d == "email":  # 3,000 emails x 25 attacked copies is slow on a CPU; a fixed sample is enough
            scams = scams.sample(min(len(scams), 400), random_state=SEED)
        out[d] = {
            attack: {
                str(rate): scam_recall(
                    model.predict_proba(attack_all(scams.text, attack, rate, seed=1), scams.domain)
                )
                for rate in RATES
            }
            for attack in ATTACKS
        }
    return out


def transformer_results(df: pd.DataFrame) -> dict:
    """Aggregate the Colab runs: mean and spread over seeds, plus single-channel runs."""
    out = {"seeds": {}, "single_channel": {}}
    key = ["split", "domain", "label", "source", "year"]
    for name in MAIN_SEEDS + ["sms_seed0", "email_seed0", "final_seed0"]:
        path = RUNS / name / "predictions.parquet"
        if not path.exists():
            print("missing run", name)
            continue
        pred = pd.read_parquet(path)
        frame = df.loc[pred.index]
        assert (
            frame[key].values == pred[key].values
        ).all(), f"{name}: predictions don't line up with the corpus"
        proba = pred[[f"p_{label}" for label in LABELS]].to_numpy()
        res = evaluate_predictions(frame, proba)
        res["history"] = json.loads((RUNS / name / "history.json").read_text())
        res["meta"] = json.loads((RUNS / name / "meta.json").read_text())
        if name in MAIN_SEEDS:
            out["seeds"][name] = res
        elif name == "final_seed0":
            out["final"] = {
                "val": {d: channel_metrics(*_val(frame, proba, d)) for d in DOMAINS},
                "meta": res["meta"],
            }
        else:
            out["single_channel"][name] = {d: res[d]["macro_f1"] for d in DOMAINS}
        out.setdefault("predictions", {})[name] = (frame, proba)
    return out


def _val(frame, proba, d):
    m = ((frame.split == "val") & (frame.domain == d) & frame.label.isin(LABELS)).to_numpy()
    return frame[m], proba[m]


def mean_over_seeds(seeds: dict) -> dict:
    """Same structure as one run, numbers replaced by the mean, plus *_std for the headline metrics."""
    runs = list(seeds.values())
    first = runs[0]

    def merge(path, node):
        if isinstance(node, dict):
            return {k: merge(path + [k], v) for k, v in node.items()}
        if isinstance(node, int | float) and not isinstance(node, bool):
            vals = []
            for r in runs:
                v = r
                for k in path:
                    v = v[k]
                vals.append(v)
            return round(float(np.mean(vals)), 4)
        return node

    out = merge([], {k: v for k, v in first.items() if k not in ("history", "meta")})
    for d in [*DOMAINS, "sms_uci_mendeley", "sms_exais"]:
        for metric in ("macro_f1", "scam_f1", "legit_false_positive_rate"):
            out[d][f"{metric}_std"] = round(float(np.std([r[d][metric] for r in runs])), 4)
    out["future"]["phishing_recall_std"] = round(
        float(np.std([r["future"]["phishing_recall"] for r in runs])), 4
    )
    out["n_seeds"] = len(runs)
    return out


def leakage_experiment() -> dict:
    """SMS only, as in v2: how much do naive merges and random splits flatter a model?"""
    uci, men = load_uci(), load_mendeley()
    naive = pd.concat([uci, men], ignore_index=True)
    naive["y"] = np.where(naive.label == "ham", "legit", "spam")
    exact = naive.assign(key=naive.text.map(dedup_key)).drop_duplicates("key")

    def run(frame, seed):
        tr, te = train_test_split(frame, test_size=0.15, stratify=frame.y, random_state=seed)
        m = TfidfLR().fit(tr.text, ["sms"] * len(tr), tr.y)
        return f1_score(te.y != "legit", scam_score(m.predict_proba(te.text, ["sms"] * len(te))) >= 0.5)

    out = {}
    for name, frame in [("naive_merge_random_split", naive), ("exact_dedup_random_split", exact)]:
        scores = [run(frame, s) for s in range(3)]
        out[name] = {"f1_mean": round(float(np.mean(scores)), 4), "f1_std": round(float(np.std(scores)), 4)}
    tr_n, te_n = train_test_split(naive, test_size=0.15, stratify=naive.y, random_state=0)
    out["naive_test_messages_with_exact_copy_in_train"] = round(
        float(te_n.text.map(dedup_key).isin(set(tr_n.text.map(dedup_key))).mean()), 3
    )
    return out


def exais_ablation(train: pd.DataFrame, scored: pd.DataFrame) -> dict:
    """ExAIS defines spam differently (carrier notices, quote subscriptions). What would training
    on it have done? Same TF-IDF model, trained without it (the protocol) and with it added.
    ExAIS is then scored on its own held-back campaigns only, to keep the comparison fair."""
    out = {}
    exais = scored[(scored.split == "external") & scored.label.isin(LABELS)]
    rng = np.random.default_rng(SEED)
    groups = exais.group.unique()
    held = set(rng.choice(groups, size=len(groups) // 3, replace=False))
    ex_train, ex_test = exais[~exais.group.isin(held)], exais[exais.group.isin(held)]
    scored = pd.concat([scored[scored.split != "external"], ex_test])
    for name, part in [("without_exais", train), ("with_exais", pd.concat([train, ex_train]))]:
        m = TfidfLR().fit(part.text, part.domain, part.label)
        res = evaluate_predictions(scored, m.predict_proba(scored.text, scored.domain))
        out[name] = {
            k: {f: res[k][f] for f in ("macro_f1", "legit_false_positive_rate", "scam_recall")}
            for k in ("sms_uci_mendeley", "sms_exais", "email")
        }
    return out


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    df = pd.read_parquet(
        PROCESSED / "corpus.parquet",
        columns=["text", "label", "domain", "source", "year", "split", "future_novel", "group"],
    )
    df["prepared"] = [prepare(t, d) for t, d in zip(df.text, df.domain, strict=True)]
    labelled = df[df.label.isin(LABELS)]
    train, val = labelled[labelled.split == "train"], labelled[labelled.split == "val"]
    scored = df[df.split.isin(["val", "test", "future", "external"])]
    results = {
        "test_size": {
            d: scored[(scored.split == "test") & (scored.domain == d)].label.value_counts().to_dict()
            for d in DOMAINS
        }
    }
    results["future_size"] = int((scored.split == "future").sum())
    results["models"], results["robustness"], results["latency_ms"], results["size_mb"] = {}, {}, {}, {}
    predictions = {}

    # -- baselines, trained here -------------------------------------------------
    for model in [NBOriginal(), TfidfLR()]:
        started = time.time()
        model.fit(train.text, train.domain, train.label)
        print(f"trained {model.name} in {time.time() - started:.0f}s", flush=True)
        path = ARTIFACTS / f"{model.name}.joblib"
        joblib.dump(model, path, compress=3)
        proba = model.predict_proba(scored.text, scored.domain)
        results["models"][model.name] = evaluate_predictions(scored, proba)
        results["latency_ms"][model.name] = latency_ms(model)
        results["size_mb"][model.name] = round(path.stat().st_size / 1e6, 1)
        results["robustness"][model.name] = robustness(model, val)
        predictions[model.name] = proba
        print(model.name, {d: results["models"][model.name][d]["macro_f1"] for d in DOMAINS}, flush=True)

    # What the normaliser buys under attack: same model, obfuscation left in
    raw = TfidfLR(deobfuscate=False).fit(train.text, train.domain, train.label)
    results["robustness"]["tfidf_lr_without_normaliser"] = robustness(raw, val)

    # Cross-channel transfer for the baseline (the transformer's comes from the Colab runs)
    results["transfer"] = {"tfidf_lr": {}}
    for name, domains in [("sms", ["sms"]), ("email", ["email"]), ("sms-email", ["sms", "email"])]:
        part = train[train.domain.isin(domains)]
        m = TfidfLR().fit(part.text, part.domain, part.label)
        proba = m.predict_proba(scored.text, scored.domain)
        res = evaluate_predictions(scored, proba)
        results["transfer"]["tfidf_lr"][name] = {d: res[d]["macro_f1"] for d in DOMAINS}

    # -- transformer, from the Colab runs ---------------------------------------
    tr = transformer_results(df)
    if tr["seeds"]:
        results["models"]["distilroberta"] = mean_over_seeds(tr["seeds"])
        results["transformer_seeds"] = {
            k: {d: v[d]["macro_f1"] for d in DOMAINS} | {"future": v["future"]["phishing_recall"]}
            for k, v in tr["seeds"].items()
        }
        results["training_history"] = {k: v["history"] for k, v in tr["seeds"].items()}
        results["transfer"]["distilroberta"] = {
            "sms": tr["single_channel"].get("sms_seed0"),
            "email": tr["single_channel"].get("email_seed0"),
            "sms-email": {d: tr["seeds"]["sms-email_seed0"][d]["macro_f1"] for d in DOMAINS},
        }
        frame, proba = tr["predictions"]["sms-email_seed0"]
        predictions["distilroberta"] = pd.DataFrame(proba, index=frame.index).reindex(scored.index).to_numpy()
    if "final" in tr:
        results["final_model"] = tr["final"]

    # Robustness and speed of the model that actually ships (int8 ONNX of the final run)
    onnx_dir = MODELS / "onnx"
    if (onnx_dir / "model.onnx").exists() and json.loads((onnx_dir / "config.json").read_text())[
        "labels"
    ] == LABELS:
        from .inference import ScamClassifier

        deployed = ScamClassifier(onnx_dir)
        results["robustness"]["distilroberta"] = robustness(deployed, val)
        results["latency_ms"]["distilroberta"] = latency_ms(deployed)
        results["size_mb"]["distilroberta"] = round((onnx_dir / "model.onnx").stat().st_size / 1e6, 1)

    results["leakage"] = leakage_experiment()
    results["exais_ablation"] = exais_ablation(train, scored)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    keep = scored[["split", "domain", "label", "source", "year", "future_novel"]].copy()
    keep["no_hook"] = ~scored.prepared.str.contains(HOOKS)
    for name, proba in predictions.items():
        for i, label in enumerate(LABELS):
            keep[f"{name}__p_{label}"] = proba[:, i]
    keep.to_parquet(REPORTS / "test_predictions.parquet")
    print("wrote reports/results.json")


if __name__ == "__main__":
    main()
