"""Upload the exported ONNX model and its model card to the Hugging Face Hub.

hf auth login                       # once, with a write token
python -m spam_detector.publish --repo hamad470/sms-scam-distilroberta
"""

import argparse
import json

from huggingface_hub import HfApi

from .paths import MODELS

CARD = """---
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
to ONNX and dynamically quantised to int8 ({size} MB).

Code, data audit and full evaluation: https://github.com/hamad470/spam-detector
Live demo: https://huggingface.co/spaces/hamad470/spam-detector

## Results

Test set: {n} messages whose spam *campaigns* (near-duplicate clusters) never appear in training.

| | value |
|---|---|
| macro-F1, 3 classes | {macro_f1:.3f} (95% CI {ci0:.3f} to {ci1:.3f}) |
| scam vs ham F1 | {bin_f1:.3f} |
| scams caught (recall) | {recall:.1%} |
| real texts wrongly flagged | {fpr:.2%} |
| smishing recall | {smish:.1%} |
| scams caught with 30% of words obfuscated | {robust:.1%} |
| CPU latency, one message | {latency} ms |

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
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="hamad470/sms-scam-distilroberta")
    args = p.parse_args()

    folder = MODELS / "onnx"
    m = json.loads((folder / "metrics.json").read_text())
    r = m["models"][m["final_model"]]
    (folder / "README.md").write_text(
        CARD.format(
            size=m["onnx"]["onnx_size_mb"],
            n=m["test_messages"],
            macro_f1=r["macro_f1"],
            ci0=r["macro_f1_ci95"][0],
            ci1=r["macro_f1_ci95"][1],
            bin_f1=r["binary"]["f1"],
            recall=r["binary"]["recall"],
            fpr=r["binary"]["ham_false_positive_rate"],
            smish=r["per_class"]["smishing"]["recall"],
            robust=r["robustness"]["mixed"]["0.3"],
            latency=m["onnx"]["latency_ms"],
        ),
        encoding="utf-8",
    )
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=folder, commit_message="Upload int8 ONNX model and card")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
