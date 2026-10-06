"""Serving-time classifier: ONNX Runtime + tokenizers, no PyTorch.

Explanations are by occlusion: drop one word, re-score, and see how far the
scam probability moves. It's model-agnostic and cheap for SMS-length text,
because every variant fits in one batch.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .decision import decide
from .text import prepare

_WORD = re.compile(r"\S+")


def _log_odds(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


@dataclass
class Result:
    label: str
    probabilities: dict[str, float]
    scam_probability: float  # P(spam) + P(smishing)
    evidence: list[dict] = field(default_factory=list)


class ScamClassifier:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text())
        self.labels = self.config["labels"]
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(self.config["max_len"])
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # free Spaces get 2 vCPUs
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )

    def _probs(self, prepared: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(prepared)
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        z = self.session.run(["logits"], feed)[0] / self.config["temperature"]
        z -= z.max(axis=1, keepdims=True)
        return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

    def predict_proba(self, texts) -> np.ndarray:
        texts = list(texts)
        return np.concatenate(
            [self._probs([prepare(t) for t in texts[i : i + 32]]) for i in range(0, len(texts), 32)]
        )

    def classify(self, text: str, explain: bool = True, top_k: int = 6) -> Result:
        p = self.predict_proba([text])[0]
        result = Result(
            label=self.labels[int(decide(p[None])[0])],
            probabilities={k: round(float(v), 4) for k, v in zip(self.labels, p, strict=True)},
            scam_probability=round(float(p[1:].sum()), 4),
        )
        if explain:
            result.evidence = self._occlusion(text, float(p[1:].sum()), top_k)
        return result

    def _occlusion(self, text: str, base: float, top_k: int) -> list[dict]:
        spans = [m.span() for m in _WORD.finditer(text)][:60]
        if len(spans) < 2:
            return []
        variants = [prepare(text[:a] + text[b:]) for a, b in spans]
        scores = self._probs(variants)[:, 1:].sum(axis=1)
        # Work in log-odds: near p=0.999 every single word looks unimportant in probability terms
        base_lo = _log_odds(base)
        evidence = [
            # positive = removing the word makes it look less like a scam, i.e. the word pointed to scam
            {"word": text[a:b], "start": a, "end": b, "impact": round(base_lo - _log_odds(float(s)), 3)}
            for (a, b), s in zip(spans, scores, strict=True)
        ]
        evidence = [e for e in evidence if abs(e["impact"]) >= 0.05]
        return sorted(evidence, key=lambda e: abs(e["impact"]), reverse=True)[:top_k]
