"""Text normalisation shared by training and inference.

Lowercase -> alphanumeric tokens -> drop English stopwords -> Porter stem.
Kept as plain functions so the fitted scikit-learn pipeline can be pickled
and reloaded anywhere the package is installed.
"""

import re
from functools import lru_cache

from nltk.stem.porter import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    from nltk.corpus import stopwords

    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens, in order."""
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=50_000)
def stem(token: str) -> str:
    return _stemmer.stem(token)


def clean_text(text: str) -> str:
    stop = _stopwords()
    return " ".join(stem(t) for t in tokenize(text) if t not in stop)


def clean_texts(texts) -> list[str]:
    """Vectorised form used inside the scikit-learn pipeline."""
    return [clean_text(t) for t in texts]
