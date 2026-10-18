# Data

Raw files aren't committed. Rebuild everything with:

```bash
python -m spam_detector.datasets        # downloads into data/raw/ (about 200 MB)
python -m spam_detector.build_dataset   # writes data/processed/corpus.parquet + reports/data_audit.json
python -m spam_detector.shortcut_audit  # writes reports/shortcut_audit.json
```

## Sources

| Dataset | Channel | Years | Raw | Unique, English | Labels | Licence |
|---|---|---|---:|---:|---|---|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (Almeida & Gómez Hidalgo, 2011) | SMS | ~2003-2011 | 5,572 | see below | ham, spam | CC BY 4.0 |
| [SMS Phishing Dataset](https://data.mendeley.com/datasets/f45bkkt8pr/1) (Mishra & Soni, 2022) | SMS | to 2022 | 5,971 | see below | ham, spam, smishing | CC BY 4.0 |
| [ExAIS_SMS](https://github.com/AbayomiAlli/SMS-Spam-Dataset) (Abayomi-Alli et al., 2022), Nigeria | SMS | 2010-2016 | 5,240 | 3,746 | ham, spam | no licence file; collected with consent for research |
| [Enron-Spam](https://huggingface.co/datasets/SetFit/enron_spam) (Metsis et al., 2006) | email | 1999-2005 | 33,716 | 30,055 | ham, spam | research use |
| [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) | email | 2002-2005 | 6,046 | 5,522 | ham, hard ham, spam | public test corpus |
| [Nazario phishing corpus](https://monkey.org/~jose/phishing/) | email | 2004-2025 | 6,197 | 4,632 | phishing | CC BY 4.0 |

UCI and Mendeley together give 5,966 unique SMS (4,921 legit, 433 spam, 562 phishing, 50 spam without a subtype).

**Unified labels:** `legit` (ham, hard ham), `spam` (unwanted marketing), `phishing` (Mendeley smishing, all
of Nazario). UCI spam that Mendeley never relabelled has no spam/phishing subtype; those 50 messages are kept
out of 3-class training and only used to check that they're flagged as scams.

**Left out:**
- `phishing3.mbox` (Nazario, 2007): Windows Defender quarantines it on download because it carries live
  malicious payloads. I'd rather lose 1 of 15 files than switch protection off.
- SmishTank (1,090 recent smishing texts): no public download.
- Hugging Face "smishing" datasets I checked were Portuguese, Hindi, very small or LLM-generated.
- 152 non-English emails (mostly Chinese phishing from 2024-2025). With no non-English legitimate email
  in the data, keeping them would teach "non-English = phishing".

## What the audit found

**1. The 2022 SMS dataset is mostly the 2011 one.** 4,943 of 5,824 unique Mendeley messages (85%) are UCI
messages, often with broken encoding (`Â£` for `£`) that hides the match from a plain comparison. Only 881
are new.

**2. Spam comes in campaigns.** MinHash clustering (5-character shingles, digits masked, Jaccard ≥ 0.7)
puts 62% of SMS phishing and 45% of email phishing in a near-duplicate cluster, against 12-15% of legitimate
messages. A random split tests on campaigns the model has already seen.

**3. Every email corpus has its own fingerprint, and a single label.** Nazario is 100% phishing; Enron and
SpamAssassin legitimate mail is 100% legit. A model can score almost perfectly by recognising the corpus:
- Enron-Spam was packaged lower-cased with a space around every punctuation mark
  (`http : / / www . removeyou . com`, `it ' s`). Nothing else looks like that.
- `jose@monkey.org`, the inbox the phishing corpus was collected from, appears in 26% of Nazario bodies.
- SpamAssassin's legitimate mail is mostly tech mailing lists (list tags like `[Spambayes]`, footers,
  "Linux", "Perl").

The fix: undo Enron's packaging at load time, then lower-case all email, strip list tags and footers, and
delete the collector's and Enron's names; addresses, links, phone numbers, amounts and years become tags.
`reports/shortcut_audit.json` measures the effect: train legit-vs-phishing on Enron + Nazario only, then score
SpamAssassin's legitimate mail, which the model never saw.

| Step | SpamAssassin legit flagged as phishing |
|---|---:|
| As downloaded | 94.7% |
| Enron packaging undone | 51.6% |
| Fingerprints scrubbed | 15.3% |

The remaining gap is a genuine topic difference (Enron was an energy company: "gas", "energy" stay strong
legitimate signals) and is listed as a limitation.

**4. Datasets disagree on what spam is.** ExAIS labels carrier service notices, Facebook birthday alerts
and motivational-quote subscriptions as spam, and some prize-draw promotions as ham
("If you don't text YES to 7070, you won't enter MTN's N2,0,000 cash draw!" is labelled ham). Training on it
made ordinary service texts (one-time codes, delivery updates) look like spam, so **ExAIS is never trained
on**: it is kept whole as an external test set.

**5. Smaller issues.**
- 7,007 exact duplicates across all sources, 87 of them with conflicting labels (the more severe label is kept).
- 66 campaign clusters contain copies labelled spam and copies labelled phishing.
- The widely used Kaggle mirror of UCI escapes quotes as `\"` and cuts 44 messages off at the first quote.
  The original UCI zip is used instead.
- Phishing header dates are often forged (1970, 2030); the year comes from Nazario's yearly file names.

## Splits

- **train / val / test** (~70/15/15, UCI/Mendeley SMS and all email): split by campaign cluster with
  `StratifiedGroupKFold`, stratified on channel and label, so no campaign appears in two splits.
- **future**: all Nazario phishing from 2023-2025 (1,192 emails), never used for training in the evaluated
  models. 1,153 of them belong to campaigns with no older relative ("future_novel").
- **external**: all of ExAIS (3,746 SMS), never trained on.

| split | channel | legit | spam | phishing | spam (no subtype) |
|---|---|---:|---:|---:|---:|
| train | SMS | 3,515 | 310 | 401 | 29 |
| train | email | 14,042 | 11,371 | 2,457 | |
| val | SMS | 703 | 61 | 81 | 7 |
| val | email | 2,808 | 2,274 | 491 | |
| test | SMS | 703 | 62 | 80 | 14 |
| test | email | 2,808 | 2,274 | 492 | |
| future | email | | | 1,192 | |
| external (ExAIS) | SMS | 2,319 | 1,427 | | |

Columns in `corpus.parquet`: `text`, `label`, `domain` (sms/email), `source`, `year`, `group` (campaign
cluster), `split`, `future_novel`, and `text_as_packaged` (Enron as downloaded, kept for the audit).
