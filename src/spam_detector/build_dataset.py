"""Merge seven sources into one de-duplicated, campaign-aware corpus for SMS and email.

    python -m spam_detector.build_dataset

Writes data/processed/corpus.parquet and reports/data_audit.json.

Labels are unified as legit / spam / phishing:
  - SMS: ham -> legit, spam -> spam, smishing -> phishing (Mendeley). UCI spam that Mendeley never
    relabelled has no subtype, so it is kept out of 3-class training ("spam_unlabelled").
  - Email: Enron/SpamAssassin ham (incl. hard ham) -> legit, their spam -> spam, Nazario -> phishing.

Splits:
  - train / val / test: ~70/15/15 by near-duplicate campaign, stratified on channel and label.
  - future: Nazario phishing from 2023-2025, never trained on. Tests whether the model catches
    phishing written after its training data. future_novel marks the ones whose campaign has no
    older relative.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .datasets import load_enron, load_exais, load_mendeley, load_nazario, load_spamassassin, load_uci
from .dedup import near_duplicate_groups
from .paths import PROCESSED, REPORTS
from .text import dedup_key

SEED = 42
FUTURE_YEARS = (2023, 2024, 2025)
SEVERITY = {"legit": 0, "spam": 1, "spam_unlabelled": 1, "phishing": 2}


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return 1.0 if not letters else sum(c.isascii() for c in letters) / len(letters)


def sms_corpus(audit: dict) -> pd.DataFrame:
    uci, men, exais = load_uci(), load_mendeley(), load_exais()
    audit["raw"] |= {"uci_2011": len(uci), "mendeley_2022": len(men), "exais_2014": len(exais)}
    uci = uci.assign(label=uci.label.map({"ham": "legit", "spam": "spam"}))
    men = men.assign(label=men.label.map({"ham": "legit", "spam": "spam", "smishing": "phishing"}))
    exais = exais.assign(label=exais.label.map({"ham": "legit", "spam": "spam"}))
    for df in (uci, men, exais):
        df["key"] = df.text.map(dedup_key)

    # Mendeley re-labels most of UCI: where they overlap, its 3-way label wins
    in_men = uci.key.isin(set(men.key))
    audit["mendeley_rows_copied_from_uci"] = int(men.key.drop_duplicates().isin(set(uci.key)).sum())
    audit["mendeley_rows_genuinely_new"] = int((~men.key.drop_duplicates().isin(set(uci.key))).sum())
    uci_only = uci[~in_men].copy()
    uci_only.loc[uci_only.label == "spam", "label"] = "spam_unlabelled"
    men["source"] = np.where(men.key.isin(set(uci.key)), "uci+mendeley", "mendeley_2022")
    return pd.concat([men, uci_only, exais], ignore_index=True)


def email_corpus(audit: dict) -> pd.DataFrame:
    enron, sa, naz = load_enron(), load_spamassassin(), load_nazario()
    audit["raw"] |= {"enron": len(enron), "spamassassin": len(sa), "nazario": len(naz)}
    enron["label"] = enron.label.map({"ham": "legit", "spam": "spam"})
    sa["label"] = sa.label.map({"ham": "legit", "hard_ham": "legit", "spam": "spam"})
    # Header dates in phishing are often forged (1970, 2030); the yearly file names are reliable
    naz["year"] = naz.file.str.extract(r"(\d{4})$")[0].astype(float).fillna(2005).astype(int)
    df = pd.concat([enron, sa, naz], ignore_index=True)
    df["key"] = df.text.map(dedup_key)
    return df


def build() -> tuple[pd.DataFrame, dict]:
    audit = {"raw": {}}
    df = pd.concat([sms_corpus(audit), email_corpus(audit)], ignore_index=True)

    english = df.text.map(latin_share) >= 0.8
    audit["dropped_non_english"] = df[~english].groupby("source").size().to_dict()
    df = df[english & (df.key.str.len() >= 8)]

    # Exact duplicates: same text with different labels is label noise; keep the more severe label
    conflicts = df.groupby(["domain", "key"]).label.nunique()
    audit["exact_duplicates"] = int(df.duplicated(["domain", "key"]).sum())
    audit["duplicates_with_conflicting_labels"] = int((conflicts > 1).sum())
    df = df.assign(sev=df.label.map(SEVERITY)).sort_values(["sev", "year"], ascending=[False, True])
    df = df.drop_duplicates(["domain", "key"]).drop(columns=["sev", "key"]).reset_index(drop=True)

    # Campaign clusters, per channel. Emails are compared on their first 1,000 characters.
    df["group"] = -1
    offset = 0
    for domain in ("sms", "email"):
        idx = df.index[df.domain == domain]
        groups = near_duplicate_groups(df.loc[idx, "text"].str.slice(0, 1000))
        df.loc[idx, "group"] = np.asarray(groups) + offset
        offset += len(idx)
    sizes = df.groupby("group").size()
    in_campaign = df.group.map(sizes) > 1
    audit["campaigns"] = {
        f"{d}_{lab}_share_in_campaign": round(
            float(in_campaign[(df.domain == d) & (df.label == lab)].mean()), 3
        )
        for d in ("sms", "email")
        for lab in ("legit", "spam", "phishing")
    }
    mixed = df[df.label.isin(["spam", "phishing"])].groupby("group").label.nunique()
    audit["campaigns_labelled_both_spam_and_phishing"] = int((mixed > 1).sum())

    df["split"] = _split(df)
    audit["final"] = {
        f"{split}/{domain}": g.label.value_counts().to_dict()
        for (split, domain), g in df.groupby(["split", "domain"])
    }
    audit["future_novel"] = int(df.future_novel.sum())
    return df, audit


def _split(df: pd.DataFrame) -> pd.Series:
    future = (df.source == "nazario") & df.year.isin(FUTURE_YEARS)
    # A future email whose campaign also has an older member is still a fair "future" test only
    # if that older member is kept out of training. Simpler and stricter: mark which future
    # emails are genuinely new campaigns, and report both numbers.
    past_groups = set(df.loc[~future, "group"])
    df["future_novel"] = future & ~df.group.isin(past_groups)

    split = pd.Series("test", index=df.index)
    split[future] = "future"
    pool = df[~future & (df.label != "spam_unlabelled")]
    strata = pool.domain + "/" + pool.label
    folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    fold_of_group = {}
    for fold, (_, idx) in enumerate(folds.split(pool, strata, pool.group)):
        for g in pool.group.iloc[idx].unique():
            fold_of_group[g] = fold
    fold = df.group.map(fold_of_group)
    keep = ~future
    split[keep & (fold >= 2)] = "train"
    split[keep & (fold == 1)] = "val"
    return split


def main() -> None:
    df, audit = build()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "corpus.parquet", index=False)
    (REPORTS / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
