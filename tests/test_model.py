import json
from pathlib import Path

import pytest

from spam_detector.predict import DEFAULT_MODEL_DIR, SpamDetector
from spam_detector.preprocess import clean_text


def test_clean_text_lowercases_stems_and_drops_stopwords():
    assert clean_text("You are WINNING the Prizes!!!") == "win prize"


def test_clean_text_handles_empty_and_symbol_only_input():
    assert clean_text("") == ""
    assert clean_text("?!... :)") == ""


@pytest.fixture(scope="module")
def detector():
    return SpamDetector()


def test_obvious_spam_and_ham(detector):
    spam = detector.predict("URGENT! You have won a £2000 cash prize. Call 09061790121 to claim now")
    ham = detector.predict("Are we still meeting for lunch tomorrow?")
    assert spam.label == "spam" and spam.spam_probability > 0.9
    assert ham.label == "ham" and ham.spam_probability < 0.1


def test_evidence_points_the_right_way(detector):
    result = detector.predict("Claim your free prize now")
    assert result.evidence, "expected at least one explaining word"
    assert all(e["word"] in "claim your free prize now" for e in result.evidence)
    assert max(e["weight"] for e in result.evidence) > 0


def test_saved_metrics_meet_quality_bar():
    card = json.loads((Path(DEFAULT_MODEL_DIR) / "metrics.json").read_text())
    m = card["test_metrics"]
    assert m["precision"] >= 0.95  # false positives (blocking real messages) are the costly error
    assert m["f1"] >= 0.9
