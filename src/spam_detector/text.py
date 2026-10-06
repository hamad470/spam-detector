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


_EMAIL_TOKEN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _deleet(token: str) -> str:
    if _EMAIL_TOKEN.search(token):  # '@' is leetspeak for 'a', but not inside an address
        return token
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
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), " xxemail "),
    (
        re.compile(r"(https?://|www\.)\S+|\b\S+\.(com|co\.uk|net|org|ly|info|biz|io|xyz|top)\b", re.I),
        " xxurl ",
    ),
    (re.compile(r"\+?\d[\d\s-]{7,}\d"), " xxphone "),
    (re.compile(r"[£$€]\s?\d[\d,.]*|\d[\d,.]*\s?([£$€]|pounds\b|gbp\b|usd\b|dollars\b)", re.I), " xxmoney "),
    (re.compile(r"\b(19[89]\d|20[0-3]\d)\b"), " xxyear "),
]


def tag_entities(text: str) -> str:
    for pattern, tag in _TAGS:
        text = pattern.sub(tag, text)
    return text


# ---- Email-specific clean-up ----------------------------------------------
# Found by the shortcut audit (reports/shortcut_audit.json): without these, a model can tell
# which corpus an email came from, and each corpus carries a single label.

# Enron-Spam ships lower-cased with spaces around punctuation ("price , only 15 $ !").
_SPACE_BEFORE = re.compile(r"\s+([,.;:!?%)\]}'])")
_SPACE_AFTER = re.compile(r"([(\[{$£€])\s+")
# Mailing-list tags and footers: SpamAssassin's legitimate mail is mostly tech mailing lists.
_LIST_TAG = re.compile(r"^(subject:\s*)((re|fwd?):\s*)*\[[^\]]{2,30}\]\s*", re.I | re.M)
_LIST_FOOTER = re.compile(
    r"^.*(mailman/listinfo|mailing list|list-unsubscribe|_{20,}|-{20,}).*$", re.I | re.M
)
# The people and organisation each corpus belongs to.
_IDENTITY = re.compile(r"\b(jose|monkey|nazario|enron|kaminski|vince|ect|hou)\b", re.I)


def scrub_email(text: str) -> str:
    text = _EMAIL_TOKEN.sub(" xxemail ", text)  # before identity masking, which would split addresses
    text = _LIST_FOOTER.sub(" ", _LIST_TAG.sub(r"\1", text))
    text = text.lower()
    text = _SPACE_AFTER.sub(r"\1", _SPACE_BEFORE.sub(r"\1", text))
    return _IDENTITY.sub(" ", text)  # deleted, not tagged: a tag would only ever appear in one corpus


def prepare(text: str, domain: str = "sms", deobfuscate: bool = True) -> str:
    """The exact string a model sees. The browser port in web/scam.js must match this.

    deobfuscate=False exists only for the ablation that measures what normalise() buys.
    """
    if deobfuscate:
        text = normalise(text)
    if domain == "email":
        text = scrub_email(text)
    return f"[{domain}] " + " ".join(tag_entities(text).split())
