"""Upload the exported ONNX model and its model card to the Hugging Face Hub.

hf auth login                       # once, with a write token
python -m spam_detector.publish     # after export, evaluate and export --card
"""

import argparse
import json

from huggingface_hub import HfApi

from .paths import MODELS, REPORTS

CARD = """---
license: mit
language: en
pipeline_tag: text-classification
library_name: onnx
base_model: distilroberta-base
tags: [sms, email, spam, phishing, smishing, onnx, int8]
---

# SMS & email scam classifier (DistilRoBERTa, int8 ONNX)

Classifies a text message or an email as **legit**, **spam** (unwanted marketing) or **phishing**
(scams, fraud, credential theft). Fine-tuned from `distilroberta-base` on about 35,000 de-duplicated
messages from six public datasets, with obfuscation augmentation (leetspeak, look-alike characters,
spaced-out letters, typos). Temperature-scaled, exported to ONNX and quantised to int8 ({size} MB).

- Code, data audit and full evaluation: https://github.com/hamad470/spam-detector
- Live demo (runs in your browser): https://huggingface.co/spaces/hamadurrehman62/spam-detector

## Results

Test sets contain only spam *campaigns* (near-duplicate clusters) that never appear in training.
Mean of {seeds} training runs.

| | SMS | Email |
|---|---|---|
| macro-F1, 3 classes | {sms_f1:.3f} | {email_f1:.3f} |
| scam vs legit F1 | {sms_scam_f1:.3f} | {email_scam_f1:.3f} |
| legitimate messages wrongly flagged | {sms_fpr:.2%} | {email_fpr:.2%} |
| phishing recall | {sms_phish:.1%} | {email_phish:.1%} |

Phishing emails from 2023-2025, written after the training data ends: **{future:.1%} flagged as scams**.

This deployed model was then retrained on the training, test and 2023-2025 data together
(validation kept aside for early stopping and calibration), so the table above describes the
protocol, not this exact checkpoint.

## Use

Input text must go through `spam_detector.text.prepare(text, "sms" | "email")` (undo
obfuscation, clean email formatting, tag links, numbers and amounts, add a channel prefix),
and logits are divided by the temperature in `config.json`. `spam_detector.inference.ScamClassifier`
does both; `web/scam.js` in the repo is a tested JavaScript port.

## Limitations

- Legitimate email in the training data is from 1999-2005 (Enron, SpamAssassin); there is no public
  modern legitimate email corpus. The report measures how much this matters and how it was mitigated.
- Scams with no link, number or amount are caught far less often (see the report's "no-hook" test).
- English only. Spam vs phishing is a soft boundary in the source labels.

## Data

UCI SMS Spam Collection; Mishra & Soni SMS Phishing Dataset (DOI 10.17632/f45bkkt8pr.1); ExAIS_SMS;
Enron-Spam (Metsis et al., 2006); SpamAssassin public corpus; Nazario phishing corpus (CC BY 4.0).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="hamadurrehman62/scam-detector-distilroberta")
    args = p.parse_args()

    folder = MODELS / "onnx"
    card = json.loads((folder / "metrics.json").read_text())
    r = json.loads((REPORTS / "results.json").read_text())["models"]["distilroberta"]
    (folder / "README.md").write_text(
        CARD.format(
            size=card["onnx"]["onnx_size_mb"],
            seeds=r["n_seeds"],
            sms_f1=r["sms"]["macro_f1"],
            email_f1=r["email"]["macro_f1"],
            sms_scam_f1=r["sms"]["scam_f1"],
            email_scam_f1=r["email"]["scam_f1"],
            sms_fpr=r["sms"]["legit_false_positive_rate"],
            email_fpr=r["email"]["legit_false_positive_rate"],
            sms_phish=r["sms"]["per_class"]["phishing"]["recall"],
            email_phish=r["email"]["per_class"]["phishing"]["recall"],
            future=r["future"]["scam_recall"],
        ),
        encoding="utf-8",
    )
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo, folder_path=folder, commit_message="Upload v3 int8 ONNX model and card"
    )
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
