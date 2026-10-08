"""Export the deployed transformer to int8 ONNX, and write the model card the demo reads.

    python -m spam_detector.export            # ONNX + quantisation + agreement check
    python -m spam_detector.evaluate          # (uses the ONNX model for robustness/latency)
    python -m spam_detector.export --card     # models/onnx/metrics.json from reports/results.json

Serving then needs onnxruntime + tokenizers (~60 MB) instead of PyTorch + transformers (~1 GB).
"""

import argparse
import json
import shutil

import numpy as np
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .decision import decide
from .models.transformer import MAX_LEN, DistilRobertaClassifier
from .paths import ARTIFACTS, DOMAINS, LABELS, MODELS, PROCESSED, REPORTS

FINAL_RUN = ARTIFACTS / "runs" / "final_seed0"
OUT = MODELS / "onnx"


def export(run_dir=FINAL_RUN, out_dir=OUT) -> None:
    clf = DistilRobertaClassifier(run_dir / "model", device="cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "model.fp32.onnx"
    sample = clf.tokenizer(["export sample"], return_tensors="pt")
    torch.onnx.export(
        clf.model,
        (sample["input_ids"], sample["attention_mask"]),
        fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask")}
        | {"logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(fp32, out_dir / "model.onnx", weight_type=QuantType.QInt8)
    fp32.unlink()

    clf.tokenizer.save_pretrained(out_dir)
    for extra in ("vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json"):
        (out_dir / extra).unlink(missing_ok=True)  # tokenizer.json alone is enough for `tokenizers`
    config = {
        "labels": LABELS,
        "temperature": clf.temperature,
        "max_len": MAX_LEN,
        "base_model": "distilroberta-base",
        "trained_on": "train + test + future splits (see reports/REPORT.md)",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy(run_dir / "history.json", out_dir / "training_history.json")


def check_agreement(run_dir=FINAL_RUN) -> dict:
    """Quantisation shouldn't change decisions. Compared on the validation split, the one
    part of the corpus the final model was never trained on."""
    from .inference import ScamClassifier

    df = pd.read_parquet(PROCESSED / "corpus.parquet")
    val = df[(df.split == "val") & df.label.isin(LABELS)]
    torch_p = DistilRobertaClassifier(run_dir / "model", device="cpu").predict_proba(val.text, val.domain)
    onnx_p = ScamClassifier(OUT).predict_proba(val.text, val.domain)
    report = {
        "decision_agreement": round(float((decide(torch_p) == decide(onnx_p)).mean()), 4),
        "max_abs_prob_diff": round(float(np.abs(torch_p - onnx_p).max()), 4),
        "median_abs_prob_diff": round(float(np.median(np.abs(torch_p - onnx_p).max(1))), 5),
        "onnx_size_mb": round((OUT / "model.onnx").stat().st_size / 1e6, 1),
        "pytorch_size_mb": round((run_dir / "model" / "model.safetensors").stat().st_size / 1e6, 1),
        "n": len(val),
    }
    (ARTIFACTS / "onnx_check.json").write_text(json.dumps(report, indent=2))
    return report


def write_card() -> None:
    """The slice of results.json the web demo shows."""
    r = json.loads((REPORTS / "results.json").read_text())
    audit = json.loads((REPORTS / "shortcut_audit.json").read_text())
    tr = r["models"]["distilroberta"]
    card = {
        "headline_model": "distilroberta",
        "n_seeds": tr["n_seeds"],
        "test_size": r["test_size"],
        "future_size": r["future_size"],
        "onnx": json.loads((ARTIFACTS / "onnx_check.json").read_text())
        | {"latency_ms": r["latency_ms"].get("distilroberta")},
        "models": {
            name: {
                d: {k: m[d][k] for k in ("macro_f1", "scam_f1", "legit_false_positive_rate")}
                | {"phishing_recall": m[d]["per_class"]["phishing"]["recall"]}
                for d in [*DOMAINS, "sms_uci_mendeley"]
            }
            | {
                "future_phishing_recall": m["future"]["phishing_recall"],
                "future_scam_recall": m["future"]["scam_recall"],
                "robust_sms_mixed_30": r["robustness"]
                .get(name, {})
                .get("sms", {})
                .get("mixed", {})
                .get("0.3"),
                "size_mb": r["size_mb"].get(name),
            }
            for name, m in r["models"].items()
        },
        "shortcut_audit": {
            k: audit[k]["flagged_share_spamassassin_legit_unseen_source"]
            for k in ("as_packaged", "raw", "scrubbed")
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(card, indent=2))
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--card", action="store_true", help="only write metrics.json (run after evaluate)")
    args = p.parse_args()
    if args.card:
        write_card()
    else:
        export()
        print(check_agreement())
