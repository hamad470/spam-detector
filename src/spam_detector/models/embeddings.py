"""Frozen sentence embeddings + logistic regression.

The middle rung: semantic features without fine-tuning anything.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..paths import LABELS
from ..text import normalise

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingLR:
    name = "minilm_lr"

    def __init__(self, encoder: str = ENCODER, C: float = 4.0):
        self.encoder_name = encoder
        self.C = C
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name, device="cpu")
        return self._encoder

    def embed(self, texts) -> np.ndarray:
        return self.encoder.encode(
            [normalise(t) for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )

    def fit(self, texts, labels):
        y = np.array([LABELS.index(label) for label in labels])
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(
            self.embed(texts), y
        )
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.embed(list(texts)))

    def __getstate__(self):  # don't pickle the encoder; it reloads from the Hub cache
        return {k: v for k, v in self.__dict__.items() if k != "_encoder"} | {"_encoder": None}
