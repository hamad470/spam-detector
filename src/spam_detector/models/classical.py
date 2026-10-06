"""Bag-of-words baselines.

nb_original  - the pipeline from my first notebook (stemmed words, TF-IDF, chi2, Bernoulli NB)
tfidf_lr     - word and character n-grams + logistic regression, with obfuscation undone first

Every model takes raw texts plus each text's channel ("sms" or "email") and sees them
through text.prepare(), so email clean-up and the channel prefix are applied the same way
for every model.
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
from ..text import prepare


def prepare_all(texts, domains, deobfuscate: bool = True) -> list[str]:
    return [prepare(t, d, deobfuscate=deobfuscate) for t, d in zip(texts, domains, strict=True)]


def encode_labels(labels) -> np.ndarray:
    return np.array([LABELS.index(label) for label in labels])


class _Base:
    name = "base"

    def predict(self, texts, domains) -> np.ndarray:
        return np.array(LABELS)[decide(self.predict_proba(texts, domains))]


class NBOriginal(_Base):
    name = "nb_original"

    def fit(self, texts, domains, labels):
        self.model = Pipeline(
            [
                ("clean", FunctionTransformer(clean_texts)),
                ("tfidf", TfidfVectorizer(max_features=3000)),
                ("select", SelectKBest(chi2, k=1000)),
                ("clf", BernoulliNB()),
            ]
        )
        self.model.fit(prepare_all(texts, domains), encode_labels(labels))
        return self

    def predict_proba(self, texts, domains):
        return self.model.predict_proba(prepare_all(texts, domains))


class TfidfLR(_Base):
    name = "tfidf_lr"

    def __init__(self, C: float = 10.0, deobfuscate: bool = True):
        self.C = C
        self.deobfuscate = deobfuscate

    def _features(self, texts, domains, fit=False):
        x = prepare_all(texts, domains, self.deobfuscate)
        if fit:
            self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)
            self.char = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=150_000
            )
            return hstack([self.word.fit_transform(x), self.char.fit_transform(x)]).tocsr()
        return hstack([self.word.transform(x), self.char.transform(x)]).tocsr()

    def fit(self, texts, domains, labels):
        X = self._features(texts, domains, fit=True)
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.clf.fit(X, encode_labels(labels))
        return self

    def predict_proba(self, texts, domains):
        return self.clf.predict_proba(self._features(texts, domains))
