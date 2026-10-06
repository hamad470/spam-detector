"""Frozen sentence embeddings + logistic regression.

The middle rung: semantic features without fine-tuning anything.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .classical import _Base, encode_labels, prepare_all

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingLR(_Base):
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
            self._encoder.max_seq_length = 256
        return self._encoder

    def embed(self, texts, domains) -> np.ndarray:
        return self.encoder.encode(
            prepare_all(texts, domains), batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )

    def fit(self, texts, domains, labels):
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.clf.fit(self.embed(texts, domains), encode_labels(labels))
        return self

    def predict_proba(self, texts, domains):
        return self.clf.predict_proba(self.embed(texts, domains))

    def __getstate__(self):  # don't pickle the encoder; it reloads from the Hub cache
        return {k: v for k, v in self.__dict__.items() if k != "_encoder"} | {"_encoder": None}
