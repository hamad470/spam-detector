"""Merge UCI (2011) and Mendeley (2022) into one de-duplicated, campaign-aware corpus.

    python -m spam_detector.build_dataset

Writes data/processed/corpus.parquet and reports/data_audit.json.

Decisions worth knowing about:
- Mendeley reuses most of UCI. Exact duplicates (after fixing encoding) are merged
  and the Mendeley label wins, because it splits UCI's "spam" into spam vs smishing.
- UCI spam that Mendeley never relabelled has no spam/smishing subtype. It is kept
  out of 3-class training and used only to test binary detection.
- Splits are made per near-duplicate cluster, so a campaign never straddles train and test.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .datasets import load_mendeley, load_uci
from .dedup import near_duplicate_groups
from .paths import PROCESSED, REPORTS
from .text import dedup_key

SEED = 42


def build() -> tuple[pd.DataFrame, dict]:
    uci, men = load_uci(), load_mendeley()
    audit = {"raw": {"uci_2011": len(uci), "mendeley_2022": len(men)}}

    uci["key"], men["key"] = uci.text.map(dedup_key), men.text.map(dedup_key)
    men = men[men.key != ""]
    conflicts = men.groupby("key").label.nunique()
    audit["mendeley_exact_duplicates"] = int(men.key.duplicated().sum())
    audit["mendeley_duplicates_with_conflicting_labels"] = int((conflicts > 1).sum())
    # Where the same text was labelled twice, keep the more severe label
    severity = {"ham": 0, "spam": 1, "smishing": 2}
    men = men.assign(sev=men.label.map(severity)).sort_values("sev", ascending=False)
    men = men.drop_duplicates("key").drop(columns="sev")

    audit["uci_exact_duplicates"] = int(uci.key.duplicated().sum())
    uci = uci.drop_duplicates("key")

    in_men = uci.key.isin(set(men.key))
    audit["mendeley_rows_copied_from_uci"] = int(men.key.isin(set(uci.key)).sum())
    audit["mendeley_rows_genuinely_new"] = int((~men.key.isin(set(uci.key))).sum())

    men["source"] = np.where(men.key.isin(set(uci.key)), "both", "mendeley_only")
    uci_only = uci[~in_men].assign(source="uci_only")
    uci_only.loc[uci_only.label == "spam", "label"] = "spam_unlabelled"

    df = pd.concat([men, uci_only], ignore_index=True)[["text", "label", "source", "key"]]
    df["group"] = near_duplicate_groups(df.text)

    sizes = df.groupby("group").size()
    in_campaign = df.group.map(sizes) > 1
    audit["near_duplicate_clusters"] = {
        "messages": len(df),
        "clusters": int(sizes.size),
        "messages_in_multi_message_clusters": int(in_campaign.sum()),
        "share_of_ham_in_clusters": round(float(in_campaign[df.label == "ham"].mean()), 3),
        "share_of_spam_or_smishing_in_clusters": round(float(in_campaign[df.label != "ham"].mean()), 3),
    }
    labelled = df[df.label.isin(["spam", "smishing"])]
    mixed = labelled.groupby("group").label.nunique()
    audit["campaign_clusters_labelled_both_spam_and_smishing"] = int((mixed > 1).sum())
    audit["campaign_clusters_with_spam_or_smishing"] = int(mixed.size)

    df["split"] = _split(df)
    audit["final"] = {
        split: df[df.split == split].label.value_counts().to_dict() for split in ["train", "val", "test"]
    }
    return df.drop(columns="key"), audit


def _split(df: pd.DataFrame) -> pd.Series:
    """~70/15/15 by cluster, stratified on label. Unlabelled spam follows its cluster, else goes to test."""
    split = pd.Series("test", index=df.index)
    lab = df[df.label != "spam_unlabelled"]
    folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    fold_of_group = {}
    for fold, (_, idx) in enumerate(folds.split(lab, lab.label, lab.group)):
        for g in lab.group.iloc[idx].unique():
            fold_of_group[g] = fold
    fold = df.group.map(fold_of_group)
    split[fold >= 2] = "train"
    split[fold == 1] = "val"
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
