"""Source classifier: can a model tell which dataset a message came from, within one label?

    python -m spam_detector.source_audit

If a TF-IDF + logistic regression model can name the source of a legitimate email, then a
scam classifier trained on single-label corpora can use the same cue as a shortcut. The
audit is run on the text as packaged (raw) and on the text every model sees (prepared,
after the v3 clean-up), within each (channel, label) cell that has more than one source.

Metric: balanced accuracy, 5-fold cross-validation grouped by near-duplicate campaign, so a
campaign never sits on both sides of a fold. Chance is 1 / number of sources.

Only train and validation rows are used; the locked test is checked out with assert_unlocked.
Results: reports/source_audit.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from .lock import assert_unlocked
from .paths import PROCESSED, REPORTS
from .text import prepare

SEED = 42
CAP_PER_SOURCE = 3000  # keeps the audit to a few minutes on a CPU; sampled with a fixed seed
TOP_FEATURES = 12


def raw_view(frame: pd.DataFrame) -> list[str]:
    """Text as the dataset shipped it: Enron's tokenised packaging where we have it."""
    raw = frame.text_as_packaged.where(frame.text_as_packaged.notna(), frame.text)
    return [f"[{d}] {t}" for t, d in zip(raw, frame.domain, strict=True)]


def prepared_view(frame: pd.DataFrame) -> list[str]:
    return [prepare(t, d) for t, d in zip(frame.text, frame.domain, strict=True)]


def model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000),
        LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"),
    )


def cv_balanced_accuracy(texts, y, groups) -> tuple[float, float]:
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in folds.split(texts, y, groups):
        m = model().fit([texts[i] for i in tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], m.predict([texts[i] for i in te])))
    return round(float(np.mean(scores)), 4), round(float(np.std(scores, ddof=1)), 4)


def top_features(texts, y, sources) -> dict:
    m = model().fit(texts, y)
    vocab = np.array(m[0].get_feature_names_out())
    coef = m[1].coef_
    if coef.shape[0] == 1:  # binary: one row, positive = second class
        return {
            sources[1]: vocab[np.argsort(coef[0])[::-1][:TOP_FEATURES]].tolist(),
            sources[0]: vocab[np.argsort(coef[0])[:TOP_FEATURES]].tolist(),
        }
    return {s: vocab[np.argsort(coef[i])[::-1][:TOP_FEATURES]].tolist() for i, s in enumerate(sources)}


def audit_cell(frame: pd.DataFrame) -> dict:
    sources = sorted(frame.source.unique())
    y = np.array([sources.index(s) for s in frame.source])
    groups = frame.group.to_numpy()
    counts = {s: int((frame.source == s).sum()) for s in sources}
    out = {"sources": counts, "chance": round(1 / len(sources), 4)}
    for name, view in (("raw", raw_view), ("prepared", prepared_view)):
        texts = view(frame)
        mean, std = cv_balanced_accuracy(texts, y, groups)
        out[name] = {"balanced_accuracy": mean, "std": std, "top_features": top_features(texts, y, sources)}
    return out


def main() -> None:
    corpus = pd.read_parquet(PROCESSED / "corpus.parquet")
    pool = corpus[corpus.split.isin(["train", "val"])]
    assert_unlocked(pool, "source audit")
    # ExAIS is an external test for the scam models, but it is a source like any other here:
    # the question is only whether its messages can be told apart from the others.
    ext = corpus[(corpus.split == "external")]
    pool = pd.concat([pool, ext])

    results = {}
    for (domain, label), cell in pool.groupby(["domain", "label"]):
        if cell.source.nunique() < 2 or label == "spam_unlabelled":
            continue
        cell = cell.groupby("source", group_keys=False).apply(
            lambda g: g.sample(min(len(g), CAP_PER_SOURCE), random_state=SEED)
        )
        key = f"{domain}/{label}"
        results[key] = audit_cell(cell)
        r = results[key]
        print(
            f"{key}: chance {r['chance']}, raw {r['raw']['balanced_accuracy']}, "
            f"prepared {r['prepared']['balanced_accuracy']}"
        )
    (REPORTS / "source_audit.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
