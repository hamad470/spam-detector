"""Inference with per-word explanations."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .preprocess import _stopwords, stem, tokenize

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass
class Prediction:
    label: str
    spam_probability: float
    # Words from the input that pushed the decision, strongest first.
    # weight > 0 pushes towards spam, < 0 towards ham.
    evidence: list[dict] = field(default_factory=list)


class SpamDetector:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, threshold: float = 0.5):
        model_dir = Path(model_dir)
        self.pipeline = joblib.load(model_dir / "spam_pipeline.joblib")
        metrics_path = model_dir / "metrics.json"
        self.card = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        self.threshold = threshold
        self._weights = self._feature_weights()

    def _feature_weights(self) -> dict[str, float]:
        """Map each selected stem to how strongly it indicates spam."""
        tfidf = self.pipeline.named_steps["tfidf"]
        mask = self.pipeline.named_steps["select"].get_support()
        names = tfidf.get_feature_names_out()[mask]
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "feature_log_prob_"):  # Naive Bayes: log-likelihood ratio
            w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        elif hasattr(clf, "coef_"):
            w = clf.coef_[0]
        else:  # tree ensembles: unsigned importance only
            w = getattr(clf, "feature_importances_", np.zeros(len(names)))
        return dict(zip(names, w.astype(float), strict=True))

    def predict(self, text: str, top_k: int = 6) -> Prediction:
        return self.predict_batch([text], top_k)[0]

    def predict_batch(self, texts: list[str], top_k: int = 6) -> list[Prediction]:
        probs = self.pipeline.predict_proba(texts)[:, 1]
        return [
            Prediction(
                label="spam" if p >= self.threshold else "ham",
                spam_probability=round(float(p), 4),
                evidence=self._explain(t, top_k),
            )
            for t, p in zip(texts, probs, strict=True)
        ]

    def _explain(self, text: str, top_k: int) -> list[dict]:
        stop = _stopwords()
        seen: dict[str, dict] = {}
        for word in tokenize(text):
            if word in stop:
                continue
            s = stem(word)
            if s in self._weights and s not in seen:
                seen[s] = {"word": word, "weight": round(self._weights[s], 3)}
        return sorted(seen.values(), key=lambda e: abs(e["weight"]), reverse=True)[:top_k]
