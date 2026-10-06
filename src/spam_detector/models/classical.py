"""Bag-of-words baselines.

nb_original  - the pipeline from my first notebook (stemmed words, TF-IDF, chi2, Bernoulli NB)
tfidf_lr     - word and character n-grams + logistic regression, with obfuscation undone first
"""

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ..decision import decide
from ..paths import LABELS
from ..preprocess import clean_texts
from ..text import prepare, tag_entities


class _Base:
    name = "base"

    def predict_proba(self, texts) -> np.ndarray:
        return self.model.predict_proba(list(texts))

    def predict(self, texts) -> np.ndarray:
        return np.array(LABELS)[decide(self.predict_proba(texts))]


class NBOriginal(_Base):
    name = "nb_original"

    def fit(self, texts, labels):
        self.model = Pipeline(
            [
                ("clean", FunctionTransformer(clean_texts)),
                ("tfidf", TfidfVectorizer(max_features=3000)),
                ("select", SelectKBest(chi2, k=1000)),
                ("clf", BernoulliNB()),
            ]
        )
        self.model.fit(list(texts), _encode(labels))
        return self


class TfidfLR(_Base):
    name = "tfidf_lr"

    def __init__(self, C: float = 10.0, normalise_input: bool = True):
        self.C = C
        self.normalise_input = normalise_input

    def _prep(self, texts):
        return [prepare(t) if self.normalise_input else tag_entities(t) for t in texts]

    def fit(self, texts, labels):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60_000
        )
        x = self._prep(texts)
        X = hstack([self.word.fit_transform(x), self.char.fit_transform(x)]).tocsr()
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.clf.fit(X, _encode(labels))
        return self

    def _features(self, texts):
        x = self._prep(texts)
        return hstack([self.word.transform(x), self.char.transform(x)]).tocsr()

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._features(list(texts)))


def _encode(labels) -> np.ndarray:
    return np.array([LABELS.index(label) for label in labels])
