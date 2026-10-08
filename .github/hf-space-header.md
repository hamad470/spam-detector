---
title: SMS & Email Scam Detector
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: true
license: mit
short_description: Legit vs spam vs phishing for SMS and email, in your browser
models:
  - hamadurrehman62/scam-detector-distilroberta
---

# SMS & Email Scam Detector

Tells legitimate messages, marketing spam and phishing apart, for texts and emails, and shows which words
drove the decision. The fine-tuned DistilRoBERTa model (int8 ONNX) runs in your browser with ONNX Runtime
Web, so the messages you check never leave your device.

Code, data audit, evaluation and report: https://github.com/hamad470/spam-detector
Model card: https://huggingface.co/hamadurrehman62/scam-detector-distilroberta
