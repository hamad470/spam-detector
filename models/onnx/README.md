---
license: mit
language: en
pipeline_tag: text-classification
library_name: onnx
base_model: distilroberta-base
tags: [sms, spam, smishing, phishing, onnx, int8]
datasets: [ucirvine/sms_spam]
---

# SMS scam classifier (DistilRoBERTa, int8 ONNX)

Classifies a text message as **ham**, **spam** (unwanted marketing) or **smishing** (SMS phishing).
Fine-tuned from `distilroberta-base` with obfuscation augmentation (leetspeak, look-alike
characters, spaced-out letters, typos), temperature-scaled on a validation set, then exported
to ONNX and dynamically quantised to int8 (82.5 MB).

Code, data audit and full evaluation: https://github.com/hamad470/spam-detector
Live demo: https://huggingface.co/spaces/hamadurrehman62/spam-detector

## Results

Test set: 851 messages whose spam *campaigns* (near-duplicate clusters) never appear in training.

| | value |
|---|---|
| macro-F1, 3 classes | 0.921 (95% CI 0.887 to 0.953) |
| scam vs ham F1 | 0.975 |
| scams caught (recall) | 97.2% |
| real texts wrongly flagged | 0.42% |
| smishing recall | 93.8% |
| scams caught with 30% of words obfuscated | 97.2% |
| CPU latency, one message | 8.62 ms |

## Use

The model expects text passed through `spam_detector.text.prepare` (undo obfuscation, tag
links, phone numbers and amounts) and logits divided by the temperature in `config.json`.
`spam_detector.inference.ScamClassifier` does both.

## Limitations

- Trained on English SMS from roughly 2003 to 2022, mostly UK and India. Newer scam formats
  (delivery fees, QR codes, crypto) are thin in the data.
- The spam vs smishing boundary is noisy in the source labels: 22 near-identical campaigns are
  labelled both ways. Treat "spam" vs "smishing" as a risk hint, and "ham" vs not-ham as the
  reliable decision.
- Doesn't transfer to email (see the out-of-domain test in the report).

## Data

UCI SMS Spam Collection (Almeida & Gómez Hidalgo, 2011) and SMS Phishing Dataset
(Mishra & Soni, 2022, DOI 10.17632/f45bkkt8pr.1), both CC BY 4.0, merged and de-duplicated.
