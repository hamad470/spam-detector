"""Text cleaning that isn't model-specific.

Three separate jobs, kept apart on purpose:
  fix_encoding  - repair mojibake in the source files ("Â£" -> "£")
  dedup_key     - aggressive key for spotting the same message across datasets
  normalise     - undo common obfuscation before a model sees the text
  prepare       - normalise, then tag links/phones/money; what every model is fed
"""

import re
import unicodedata

import ftfy

# Characters that look Latin but aren't. Spammers swap these in to dodge keyword filters.
_CONFUSABLES = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
        "С": "C", "Т": "T", "Х": "X", "ο": "o", "α": "a", "ν": "v",
    }
)  # fmt: skip
# Only applied inside words that also contain letters, so phone numbers survive.
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_WORDISH = re.compile(r"\S+")
# c.l.a.i.m / F R E E / p-r-i-z-e. One separator per run, so "U.R.G.E.N.T Y*o*u*r" stays two words.
_SPACED_OUT = re.compile(r"\b[A-Za-z]([.\-_* ])[A-Za-z](?:\1[A-Za-z])+\b(?![.\-_*][A-Za-z])")
_ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")


def fix_encoding(text: str) -> str:
    return ftfy.fix_text(str(text)).strip()


def dedup_key(text: str) -> str:
    """Lowercase letters and digits only, so encoding junk and spacing don't hide a duplicate."""
    return re.sub(r"[^a-z0-9]", "", fix_encoding(text).lower())


def _deleet(token: str) -> str:
    has_alpha = any(c.isalpha() for c in token)
    mostly_digits = sum(c.isdigit() for c in token) > len(token) / 2
    return token.translate(_LEET) if has_alpha and not mostly_digits else token


def normalise(text: str) -> str:
    """Map obfuscated text back towards what a human reads.

    fr33 pr1ze -> free prize, c.l.a.i.m -> claim, Cyrillic 'а' -> Latin 'a'.
    """
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    text = text.translate(_CONFUSABLES)
    text = _SPACED_OUT.sub(lambda m: re.sub(r"[.\-_* ]", "", m.group()), text)
    return _WORDISH.sub(lambda m: _deleet(m.group()), text)


# Scam texts lean on links, phone numbers and money. Tagging them gives every
# model a feature that doesn't depend on the exact number or domain.
_TAGS = [
    (re.compile(r"(https?://|www\.)\S+|\b\S+\.(com|co\.uk|net|ly|info|biz)\b", re.I), " xxurl "),
    (re.compile(r"\+?\d[\d\s-]{7,}\d"), " xxphone "),
    (re.compile(r"[£$€]\s?\d[\d,.]*|\d[\d,.]*\s?(pounds|gbp|usd)\b", re.I), " xxmoney "),
]


def tag_entities(text: str) -> str:
    for pattern, tag in _TAGS:
        text = pattern.sub(tag, text)
    return text


def prepare(text: str) -> str:
    return tag_entities(normalise(text))
