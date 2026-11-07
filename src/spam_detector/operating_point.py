"""Operating points: how the scam score becomes a product decision.

A filter is judged at a fixed scam recall, not at an arbitrary 0.5 cut-off:
"if we must catch 95% of scams, how many genuine messages do we flag?"

Every threshold here is chosen on the validation split and then applied, unchanged,
to test data. `pick_threshold` never sees test labels; the caller is responsible for
passing validation rows only (`spam_detector.lock` checks this for the locked test).

Bands (product framing):
  allow   score < warn threshold                  delivered normally
  warn    warn threshold <= score < flag threshold   needs a second check
  flag    score >= flag threshold                 blocked or marked as a scam
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BANDS = ("allow", "warn", "flag")


def _as_arrays(is_scam, score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(is_scam, dtype=bool)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("is_scam and score must be 1-D arrays of the same length")
    if not y.any():
        raise ValueError("need at least one scam to set a recall target")
    return y, s


def pick_threshold(is_scam, score, target_recall: float) -> float:
    """Highest threshold whose scam recall is at least `target_recall`.

    Flagging rule is `score >= threshold`. Taking the highest such threshold gives the
    fewest false alarms that still meets the recall target on this data.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    y, s = _as_arrays(is_scam, score)
    scam_scores = np.sort(s[y])[::-1]  # descending
    # Flagging the k highest-scoring scams needs threshold <= the k-th score.
    k = int(np.ceil(target_recall * len(scam_scores)))
    return float(scam_scores[k - 1])


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    recall: float
    precision: float
    false_alarms_per_1000: float  # genuine messages flagged, per 1,000 genuine messages
    n_scam: int
    n_legit: int


def at_threshold(is_scam, score, threshold: float) -> OperatingPoint:
    y, s = _as_arrays(is_scam, score)
    flagged = s >= threshold
    n_legit = int((~y).sum())
    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    return OperatingPoint(
        threshold=round(float(threshold), 6),
        recall=round(tp / int(y.sum()), 4),
        precision=round(tp / (tp + fp), 4) if tp + fp else 0.0,
        false_alarms_per_1000=round(1000 * fp / n_legit, 2) if n_legit else float("nan"),
        n_scam=int(y.sum()),
        n_legit=n_legit,
    )


def evaluate_at_recall(val_is_scam, val_score, test_is_scam, test_score, target_recall: float = 0.95) -> dict:
    """Threshold set on validation, reported on test. Also reports the test-tuned
    ("oracle") point, labelled as such, so the cost of not peeking is visible."""
    t = pick_threshold(val_is_scam, val_score, target_recall)
    oracle = pick_threshold(test_is_scam, test_score, target_recall)
    return {
        "target_recall": target_recall,
        "from_validation": asdict(at_threshold(test_is_scam, test_score, t)),
        "oracle_not_for_reporting": asdict(at_threshold(test_is_scam, test_score, oracle)),
    }


@dataclass(frozen=True)
class Bands:
    warn: float  # lower edge of the warn band
    flag: float  # lower edge of the flag band

    def __post_init__(self) -> None:
        if not 0.0 <= self.warn <= self.flag <= 1.0:
            raise ValueError("need 0 <= warn <= flag <= 1")

    def assign(self, score) -> np.ndarray:
        s = np.asarray(score, dtype=float)
        return np.where(s >= self.flag, 2, np.where(s >= self.warn, 1, 0))

    def label(self, score: float) -> str:
        return BANDS[int(self.assign(np.array([score]))[0])]


def fit_bands(is_scam, score, warn_recall: float = 0.99, flag_recall: float = 0.95) -> Bands:
    """warn edge catches `warn_recall` of validation scams, flag edge `flag_recall`."""
    if warn_recall < flag_recall:
        raise ValueError("warn_recall must be >= flag_recall")
    return Bands(
        warn=pick_threshold(is_scam, score, warn_recall),
        flag=pick_threshold(is_scam, score, flag_recall),
    )


def band_summary(is_scam, score, bands: Bands) -> dict:
    """Share of messages in each band, and what each band holds."""
    y, s = _as_arrays(is_scam, score)
    b = bands.assign(s)
    out = {"warn_edge": bands.warn, "flag_edge": bands.flag}
    for i, name in enumerate(BANDS):
        m = b == i
        out[name] = {
            "share_of_messages": round(float(m.mean()), 4),
            "scams": int((m & y).sum()),
            "genuine": int((m & ~y).sum()),
        }
    return out
