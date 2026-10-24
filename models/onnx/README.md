---
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
spaced-out letters, typos). Temperature-scaled, exported to ONNX and quantised to int8 (82.5 MB).

- Code, data audit and full evaluation: https://github.com/hamad470/spam-detector
- Live demo (runs in your browser): https://huggingface.co/spaces/hamadurrehman62/spam-detector

## Results

Test sets contain only spam *campaigns* (near-duplicate clusters) that never appear in training.
Mean of 3 training runs.

| | SMS | Email |
|---|---|---|
| macro-F1, 3 classes | 0.857 | 0.969 |
| scam vs legit F1 | 0.968 | 0.986 |
| legitimate messages wrongly flagged | 0.23% | 0.73% |
| phishing recall | 87.9% | 95.5% |

Phishing emails from 2023-2025, written after the training data ends: **95.1% flagged as scams**.

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
