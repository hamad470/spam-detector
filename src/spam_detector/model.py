"""Model definitions: every candidate shares the same text pipeline."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .preprocess import clean_texts

CANDIDATES = {
    "bernoulli_nb": lambda: BernoulliNB(),
    "multinomial_nb": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, C=10),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}


def build_pipeline(classifier_name: str, max_features: int = 3000, k_best: int = 1000) -> Pipeline:
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_texts)),
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("select", SelectKBest(chi2, k=k_best)),
            ("clf", CANDIDATES[classifier_name]()),
        ]
    )
