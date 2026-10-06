# Data

Raw files aren't committed. Rebuild everything with:

```bash
python -m spam_detector.datasets        # downloads into data/raw/
python -m spam_detector.build_dataset   # writes data/processed/corpus.parquet + reports/data_audit.json
```

## Sources

| Dataset | Messages | Labels | Licence | Used for |
|---|---:|---|---|---|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (Almeida & Gómez Hidalgo, 2011) | 5,572 | ham, spam | CC BY 4.0 | training and test |
| [SMS Phishing Dataset](https://data.mendeley.com/datasets/f45bkkt8pr/1) (Mishra & Soni, 2022) | 5,971 | ham, spam, smishing | CC BY 4.0 | training and test |
| [Enron-Spam](https://huggingface.co/datasets/SetFit/enron_spam) (Metsis et al., 2006), test split | 2,000 | ham, spam | research use | out-of-domain test only |

## What the audit found

I expected the 2022 dataset to add a few thousand modern messages. It mostly doesn't.

- **4,942 of the 5,823 unique Mendeley messages (85%) are UCI messages**, often with broken
  encoding (`Â£` for `£`, `Ã¼` for `ü`) that hides the match from a plain string comparison.
  Only **881 messages are genuinely new**. Matching on letters and digits after repairing
  the encoding with `ftfy` is what exposed this.
- **Spam is templated.** Clustering with MinHash (5-character shingles, digits masked,
  Jaccard ≥ 0.7) puts **51% of spam/smishing messages in a multi-message campaign**, against
  7% of ham. A random split therefore tests on campaigns the model has already seen.
- **The spam/smishing boundary is noisy.** 35 messages appear twice in Mendeley with different
  labels, and 23 campaign clusters contain copies labelled spam *and* copies labelled smishing.
  Where an exact duplicate had two labels I kept the more severe one.
- **UCI has 447 exact duplicates** of its own, removed before anything else.
- **The popular Kaggle mirror of UCI is damaged.** It escapes quotes as `\"` and cuts 44
  messages off at the first quote character. This project downloads the original UCI zip instead.

## How the corpus is built

1. Repair encoding, drop exact duplicates within each source.
2. Merge. For a message in both sources the Mendeley label wins, because it splits UCI's
   "spam" into spam vs smishing.
3. UCI spam that Mendeley never relabelled (50 messages) has no subtype. It is excluded from
   3-class training and only used to check that it gets flagged as a scam at all.
4. Cluster near-duplicates, then split **by cluster** with `StratifiedGroupKFold`
   (roughly 70 / 15 / 15), so a campaign sits entirely in train, validation or test.

## Final corpus

| split | ham | spam | smishing | spam (no subtype) |
|---|---:|---:|---:|---:|
| train | 3,543 | 309 | 402 | 25 |
| validation | 709 | 62 | 80 | 6 |
| test | 709 | 62 | 80 | 19 |

Columns in `corpus.parquet`: `text`, `label`, `source` (`both` / `mendeley_only` / `uci_only`),
`group` (campaign cluster id), `split`.
