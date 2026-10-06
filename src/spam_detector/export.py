"""Export the fine-tuned transformer to int8 ONNX for CPU serving.

    python -m spam_detector.export --model distilroberta_aug

The serving image then needs onnxruntime + tokenizers (~60 MB) instead of
PyTorch + transformers (~1 GB), and inference gets roughly 2-3x faster.
"""

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .models.transformer import DistilRobertaClassifier
from .paths import ARTIFACTS, LABELS, MODELS, PROCESSED, REPORTS


def export(name: str, out_dir=MODELS / "onnx") -> dict:
    clf = DistilRobertaClassifier(ARTIFACTS / name)
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
    int8 = out_dir / "model.onnx"
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    fp32.unlink()

    clf.tokenizer.save_pretrained(out_dir)
    for extra in ("vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json"):
        (out_dir / extra).unlink(missing_ok=True)  # tokenizer.json alone is enough for `tokenizers`
    config = {
        "labels": LABELS,
        "temperature": clf.temperature,
        "max_len": clf.max_len,
        "base_model": "distilroberta-base",
        "trained_variant": name,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return config


def check_agreement(name: str) -> dict:
    """Quantisation shouldn't change decisions. Compare against the PyTorch model on the test split."""
    from .inference import ScamClassifier

    df = pd.read_parquet(PROCESSED / "corpus.parquet")
    texts = df[(df.split == "test") & df.label.isin(LABELS)].text.tolist()
    torch_p = DistilRobertaClassifier(ARTIFACTS / name).predict_proba(texts)
    onnx_p = ScamClassifier(MODELS / "onnx").predict_proba(texts)
    return {
        "argmax_agreement": round(float((torch_p.argmax(1) == onnx_p.argmax(1)).mean()), 4),
        "max_abs_prob_diff": round(float(np.abs(torch_p - onnx_p).max()), 4),
        "onnx_size_mb": round((MODELS / "onnx" / "model.onnx").stat().st_size / 1e6, 1),
    }


def write_card(name: str, onnx_report: dict) -> None:
    """The slice of results.json the web app shows, plus the ONNX numbers."""
    from .inference import ScamClassifier

    results = json.loads((REPORTS / "results.json").read_text())
    clf = ScamClassifier(MODELS / "onnx")
    text = "URGENT! Your parcel is held. Pay the £1.99 fee at royalmail-redelivery.com"
    clf.predict_proba([text])
    started = time.perf_counter()
    for _ in range(30):
        clf.predict_proba([text])
    onnx_report["latency_ms"] = round((time.perf_counter() - started) / 30 * 1000, 2)
    keep = ("macro_f1", "macro_f1_ci95", "per_class", "binary", "robustness", "size_mb")
    card = {
        "final_model": name,
        "test_messages": sum(results["test_size"].values()),
        "onnx": onnx_report,
        "models": {k: {f: v[f] for f in keep} for k, v in results["models"].items()},
        "leakage": results["leakage"],
    }
    (MODELS / "onnx" / "metrics.json").write_text(json.dumps(card, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilroberta_aug")
    args = p.parse_args()
    export(args.model)
    report = check_agreement(args.model)
    print(report)
    shutil.copy(ARTIFACTS / args.model / "history.json", MODELS / "onnx" / "training_history.json")
    write_card(args.model, report)
