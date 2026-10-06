"""Character-level obfuscations real spammers use to slip past keyword filters.

Each attack takes a rate in [0, 1]: the share of eligible words it touches.
Used for two things: measuring robustness, and augmenting training data.
"""

import random
import re

_LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
_HOMOGLYPH = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}  # Cyrillic
_WORD = re.compile(r"[A-Za-z]{3,}")


def _apply(text: str, rate: float, rng: random.Random, fn) -> str:
    return _WORD.sub(lambda m: fn(m.group(), rng) if rng.random() < rate else m.group(), text)


def leetspeak(text, rate, rng):
    return _apply(text, rate, rng, lambda w, r: "".join(_LEET.get(c.lower(), c) for c in w))


def homoglyph(text, rate, rng):
    return _apply(text, rate, rng, lambda w, r: "".join(_HOMOGLYPH.get(c, c) for c in w))


def spacing(text, rate, rng):
    return _apply(text, rate, rng, lambda w, r: r.choice([".", " ", "-", "*"]).join(w))


def typo(text, rate, rng):
    def swap(w, r):
        i = r.randrange(len(w) - 1)
        return w[:i] + w[i + 1] + w[i] + w[i + 2 :]

    return _apply(text, rate, rng, swap)


ATTACKS = {"leetspeak": leetspeak, "homoglyph": homoglyph, "spacing": spacing, "typo": typo}


def mixed(text: str, rate: float, rng: random.Random) -> str:
    """Each touched word gets a random attack, like a spammer mixing tricks."""
    fns = list(ATTACKS.values())
    return _WORD.sub(
        lambda m: rng.choice(fns)(m.group(), 1.0, rng) if rng.random() < rate else m.group(), text
    )


def attack_all(texts, name: str, rate: float, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    fn = mixed if name == "mixed" else ATTACKS[name]
    return [fn(t, rate, rng) for t in texts]
