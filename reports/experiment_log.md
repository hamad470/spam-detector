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
