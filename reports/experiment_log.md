# Experiment log (v4)

Every change is logged here, kept or dropped, with the reason. Failed experiments stay in the log.
Numbers come only from code that was run; each row names the results file or run it came from.

## Locked test set

Sealed on 7 Nov 2026 by `python -m spam_detector.lock --write` from `data/processed/corpus.parquet`
(49,921 messages, split counts identical to the committed `reports/data_audit.json`).

| | Messages |
|---|---:|
| Email: legit / spam / phishing | 2,808 / 2,274 / 492 |
| SMS: legit / spam / phishing / spam (unlabelled type) | 703 / 62 / 80 / 14 |
| **Total** | **6,433** |

Manifest: `reports/locked_test.json`, digest `cdb409ac5dff...`. `--verify` confirms no locked message
appears, after de-duplication normalisation, in the train, validation, future or external splits.

**History, stated plainly.** This is the v3 test split. Before it was sealed, v3 had already:

- reported results on it repeatedly (README and report, October 2026);
- used it in one design decision: the ExAIS ablation on the final split (report section 4) helped decide
  to keep ExAIS out of training;
- trained the deployed v3 model on it (`final_seed0`, train + test + future).

So it is not a never-seen test set for v3 decisions. From v4 onwards nothing is tuned, chosen or prompted
on it, and the v3 deployed checkpoint is not scored on it. New held-out sources (task 3) will give a
second test that no version has seen.

## Reference models

| Run | Trained on | Train size | Temperature | Locked test used for training? |
|---|---|---:|---:|---|
| `sms-email_seed0` | train | 32,096 | 1.914 | No |
| `sms-email_seed1` | train | 32,096 | 1.914 | No |
| `sms-email_seed2` | train | 32,096 | 1.905 | No |
| `final_seed0` (v3 deployed) | train + test + future | 39,707 | 1.653 | **Yes: excluded from locked-test scoring** |

Source: each run's `meta.json` from the v3 Colab runs.

## Log

| # | Date | Change | Result | Keep or drop | Why |
|---|---|---|---|---|---|
| 0 | 2026-11-07 | Seal the locked test; add operating-point metrics and bands (`operating_point.py`) | Tooling only, no model numbers | Keep | Ground rule: locked test, thresholds from validation only |
| 1 | 2026-11-07 | Reference: v3 train-only DistilRoBERTa (3 seeds) scored on the locked test, thresholds for 95% recall set on validation | SMS: 93.7% ± 1.4 recall, 1.42 false alarms per 1,000. Email: 94.6% ± 1.3 recall, 1.54 ± 1.15 per 1,000 | Keep as the v4 baseline | Every later experiment is compared with this on validation; source `reports/final_test/reference.json` |

### Notes on entry 1

- **Thresholds set on validation land slightly under 95% on test** (SMS 92.3% to 95.1%, email 93.1% to
  95.5% across seeds). That is the honest cost of not tuning on test, and it is reported, not corrected.
- **SMS false alarms are one message.** 1.42 per 1,000 is 1 of 703 genuine SMS in every seed. The SMS
  legit test set is too small to separate models on false alarms; a modern SMS ham source (task 3) is needed.
- **SpamAssassin is the weak source.** Scam recall there ranges from 73.8% to 91.3% across seeds and
  false alarms from 0 to 7.4 per 1,000: the shortcut-audit gap from v3 is still visible per source.
- **Seed spread is large in thresholds** (SMS 0.69 to 0.93), so single-seed comparisons in later
  experiments would be noise. Experiments are run on at least 3 seeds.
- **About 100 false positives for task 8 do not exist on the clean test:** at this operating point the
  locked test has 1 SMS and 2 to 8 email false alarms per seed. Task 8 will draw its false positives from
  validation, the warn band and the out-of-domain sets instead, and say so.
