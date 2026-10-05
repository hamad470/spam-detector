"""Train, compare and export the spam classifier.

    python -m spam_detector.train            # compare candidates, save the best
    python -m spam_detector.train --model bernoulli_nb

Candidates are compared with stratified 5-fold cross-validation on the
training split only. The chosen model is then scored once on a held-out
test split, and those test metrics are what the README and the app report.
"""

import argparse
import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import __version__
from .data import DEFAULT_DATA_PATH, load_dataset
from .model import CANDIDATES, build_pipeline

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SEED = 42
SCORING = ["accuracy", "precision", "recall", "f1"]


def compare_candidates(X, y) -> dict[str, dict[str, float]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name in CANDIDATES:
        scores = cross_validate(build_pipeline(name), X, y, cv=cv, scoring=SCORING, n_jobs=1)
        results[name] = {m: round(float(np.mean(scores[f"test_{m}"])), 4) for m in SCORING}
        print(f"{name:<22}" + "  ".join(f"{m}={results[name][m]:.4f}" for m in SCORING))
    return results


def evaluate(pipeline, X_test, y_test) -> dict:
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, type=Path)
    parser.add_argument("--model", choices=list(CANDIDATES), help="skip selection and train this model")
    parser.add_argument("--out", default=MODELS_DIR, type=Path)
    args = parser.parse_args(argv)

    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED
    )
    print(f"{len(df)} messages ({df['label'].mean():.1%} spam); train={len(X_train)} test={len(X_test)}\n")

    print("5-fold CV on the training split:")
    cv_results = compare_candidates(X_train, y_train)
    chosen = args.model or max(cv_results, key=lambda n: cv_results[n]["f1"])
    print(f"\nSelected: {chosen}")

    pipeline = build_pipeline(chosen).fit(X_train, y_train)
    test_metrics = evaluate(pipeline, X_test, y_test)
    print("Held-out test:", json.dumps(test_metrics))

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out / "spam_pipeline.joblib", compress=3)
    card = {
        "model": chosen,
        "package_version": __version__,
        "trained_on": date.today().isoformat(),
        "dataset": {
            "name": "UCI SMS Spam Collection",
            "messages": int(len(df)),
            "spam_share": round(float(df["label"].mean()), 4),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "test_metrics": test_metrics,
        "cv_comparison": cv_results,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    (args.out / "metrics.json").write_text(json.dumps(card, indent=2))
    print(f"Saved model and metrics to {args.out}")


if __name__ == "__main__":
    main()
