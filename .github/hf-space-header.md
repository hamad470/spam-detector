---
title: SMS Scam Detector
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: true
license: mit
short_description: Ham vs spam vs smishing, runs entirely in your browser
models:
  - hamadurrehman62/sms-scam-distilroberta
---

# SMS Scam Detector

Tells normal texts, marketing spam and smishing (SMS phishing) apart, and shows which words drove the decision.
The fine-tuned DistilRoBERTa model (int8 ONNX, 82.5 MB) runs in your browser with ONNX Runtime Web, so the
messages you check never leave your device.

Code, data audit, evaluation and report: https://github.com/hamad470/spam-detector
Model card: https://huggingface.co/hamadurrehman62/sms-scam-distilroberta
