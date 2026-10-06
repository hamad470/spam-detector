"""How three probabilities become one label.

Plain argmax can call a message "ham" while P(spam) + P(smishing) is above 0.5,
because the scam mass is split across two classes. A filter's first question is
"scam or not?", so decide that first, then which kind.
"""

import numpy as np

THRESHOLD = 0.5


def decide(proba: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """(n, 3) probabilities in LABELS order -> class index per row."""
    proba = np.asarray(proba)
    scam = 1.0 - proba[:, 0] >= threshold
    kind = 1 + proba[:, 1:].argmax(axis=1)
    return np.where(scam, kind, 0)
