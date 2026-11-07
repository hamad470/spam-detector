"""Score a trained run on the locked test set. The only module that reads locked labels.

    python -m spam_detector.final_test --runs sms-email_seed0 sms-email_seed1 sms-email_seed2 --tag reference

For each run and channel:
  1. thresholds come from that run's VALIDATION predictions only (95% scam recall), and the
     allow / warn / flag bands likewise (99% and 95% recall);
  2. they are applied unchanged to the locked test;
  3. the result is written to reports/final_test/<tag>.json, and one line per scoring is appended
     to reports/final_test/scoring_log.jsonl, so every look at the locked test is on record.

Runs trained on the locked split (v3's final_seed0) are refused.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .evaluate import RUNS, expected_calibration_error, scam_score
from .lock import LOCKED_SPLITS, verify
from .operating_point import band_summary, evaluate_at_recall, fit_bands
from .paths import DOMAINS, LABELS, PROCESSED, REPORTS

OUT = REPORTS / "final_test"
TARGET_RECALL = 0.95
KEY = ["split", "domain", "label", "source", "year"]


def load_run(name: str, corpus: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict]:
    meta = json.loads((RUNS / name / "meta.json").read_text())
    if meta.get("final"):
        raise SystemExit(f"{name} was trained on the locked split; it cannot be scored on it")
    pred = pd.read_parquet(RUNS / name / "predictions.parquet")
    frame = corpus.loc[pred.index]
    if not (frame[KEY].values == pred[KEY].values).all():
        raise SystemExit(f"{name}: predictions don't line up with the corpus")
    return frame, pred[[f"p_{label}" for label in LABELS]].to_numpy(), meta


def score_channel(frame: pd.DataFrame, proba: np.ndarray, domain: str) -> dict:
    labelled = frame.label.isin(LABELS).to_numpy()
    in_domain = (frame.domain == domain).to_numpy()
    val = labelled & in_domain & (frame.split == "val").to_numpy()
    test = labelled & in_domain & frame.split.isin(LOCKED_SPLITS).to_numpy()
    score = scam_score(proba)
    is_scam = (frame.label != "legit").to_numpy()

    point = evaluate_at_recall(is_scam[val], score[val], is_scam[test], score[test], TARGET_RECALL)
    bands = fit_bands(is_scam[val], score[val], warn_recall=0.99, flag_recall=TARGET_RECALL)
    by_source = {}
    for src in sorted(frame[test].source.unique()):
        m = test & (frame.source == src).to_numpy()
        t = point["from_validation"]["threshold"]
        flagged = score[m] >= t
        y = is_scam[m]
        by_source[src] = {
            "n": int(m.sum()),
            "scam_recall": round(float(flagged[y].mean()), 4) if y.any() else None,
            "false_alarms_per_1000": round(1000 * float(flagged[~y].mean()), 2) if (~y).any() else None,
        }
    return {
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "at_95_recall": point,
        "pr_auc": round(float(average_precision_score(is_scam[test], score[test])), 4),
        "ece": round(expected_calibration_error(is_scam[test].astype(float), score[test]), 4),
        "bands": band_summary(is_scam[test], score[test], bands),
        "by_source": by_source,
    }


def mean_std(values) -> dict:
    a = np.array([v for v in values if v is not None], dtype=float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}


def summarise(per_run: dict) -> dict:
    out = {}
    for d in DOMAINS:
        runs = [r[d] for r in per_run.values()]
        out[d] = {
            "scam_recall": mean_std(r["at_95_recall"]["from_validation"]["recall"] for r in runs),
            "precision": mean_std(r["at_95_recall"]["from_validation"]["precision"] for r in runs),
            "false_alarms_per_1000": mean_std(
                r["at_95_recall"]["from_validation"]["false_alarms_per_1000"] for r in runs
            ),
            "pr_auc": mean_std(r["pr_auc"] for r in runs),
            "ece": mean_std(r["ece"] for r in runs),
            "warn_band_share": mean_std(r["bands"]["warn"]["share_of_messages"] for r in runs),
        }
    return out


def git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--tag", required=True, help="name of this scoring, e.g. reference")
    args = p.parse_args()

    corpus = pd.read_parquet(PROCESSED / "corpus.parquet")
    verify(corpus)  # the locked split is exactly the sealed one, and nothing leaked out of it

    per_run, metas = {}, {}
    for name in args.runs:
        frame, proba, metas[name] = load_run(name, corpus)
        per_run[name] = {d: score_channel(frame, proba, d) for d in DOMAINS}

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "tag": args.tag,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "target_recall": TARGET_RECALL,
        "thresholds_from": "validation split of each run",
        "runs": metas,
        "summary": summarise(per_run),
        "per_run": per_run,
    }
    path = OUT / f"{args.tag}.json"
    if path.exists():
        raise SystemExit(f"{path} exists: each tag is scored once. Use a new tag and log why.")
    path.write_text(json.dumps(result, indent=2))
    with (OUT / "scoring_log.jsonl").open("a") as log:
        entry = {k: result[k] for k in ("tag", "scored_at", "git_commit")} | {"runs": args.runs}
        log.write(json.dumps(entry) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
