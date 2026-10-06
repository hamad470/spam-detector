"""Can a model score well on email by recognising the corpus instead of the scam?

    python -m spam_detector.shortcut_audit

Every email corpus here has one label: Nazario is all phishing, Enron and SpamAssassin
ham is all legit. So a model can "detect phishing" by spotting Nazario's fingerprints
(the collector's address, mixed case) or Enron's (lower-cased, spaced punctuation).

The test: train legit-vs-phishing on Enron ham + Nazario only, then score SpamAssassin ham,
a corpus the model never saw. A model that learned "Enron-looking = legit" will flag
SpamAssassin's normal emails as phishing. Three runs: Enron as packaged, Enron with its
packaging undone ("raw"), and everything through prepare() ("scrubbed").
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .paths import PROCESSED, REPORTS
from .text import prepare


def run(texts_train, y_train, eval_sets: dict, lowercase: bool) -> dict:
    vec = TfidfVectorizer(
        lowercase=lowercase, token_pattern=r"\S+", ngram_range=(1, 2), min_df=3, max_features=200_000
    )
    clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
    clf.fit(vec.fit_transform(texts_train), y_train)
    out = {name: float((clf.predict(vec.transform(t)) == 1).mean()) for name, t in eval_sets.items()}
    names, w = vec.get_feature_names_out(), clf.coef_[0]
    out["top_phishing_features"] = names[np.argsort(-w)[:15]].tolist()
    out["top_legit_features"] = names[np.argsort(w)[:15]].tolist()
    return out


def main() -> None:
    df = pd.read_parquet(PROCESSED / "corpus.parquet")
    email = df[df.domain == "email"]
    train = email[
        (email.split == "train")
        & ((email.source == "enron") & (email.label == "legit") | (email.source == "nazario"))
    ]
    test = email[email.split == "test"]
    sets = {
        "flagged_share_enron_legit_seen_source": test[
            (test.source == "enron") & (test.label == "legit")
        ].text,
        "flagged_share_spamassassin_legit_unseen_source": test[
            (test.source == "spamassassin") & (test.label == "legit")
        ].text,
        "caught_share_nazario_phishing": test[test.source == "nazario"].text,
    }
    y = (train.label == "phishing").astype(int).to_numpy()

    def packaged(frame):  # Enron exactly as downloaded, everything else raw
        return frame.text_as_packaged.where(frame.text_as_packaged.notna(), frame.text)

    results = {
        "as_packaged": run(
            packaged(train), y, {k: packaged(test.loc[v.index]) for k, v in sets.items()}, lowercase=False
        ),
        "raw": run(train.text, y, sets, lowercase=False),
        "scrubbed": run(
            train.text.map(lambda t: prepare(t, "email")),
            y,
            {k: v.map(lambda t: prepare(t, "email")) for k, v in sets.items()},
            lowercase=False,
        ),
        "train_size": len(train),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "shortcut_audit.json").write_text(json.dumps(results, indent=2))
    for k in ("as_packaged", "raw", "scrubbed"):
        r = results[k]
        print(f"\n{k}: " + "  ".join(f"{m}={r[m]:.3f}" for m in sets))
        print("  phishing:", r["top_phishing_features"][:10])
        print("  legit:   ", r["top_legit_features"][:10])


if __name__ == "__main__":
    main()
