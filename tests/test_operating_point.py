"""Operating-point maths and the locked-test guard, on small hand-checkable cases."""

import numpy as np
import pandas as pd
import pytest

from spam_detector.lock import LockedTestLeak, assert_unlocked, load_keys, verify, write_manifest
from spam_detector.operating_point import (
    Bands,
    at_threshold,
    band_summary,
    evaluate_at_recall,
    fit_bands,
    pick_threshold,
)

# 10 scams scored 0.10..1.00, 10 genuine scored 0.05..0.50
SCAM = np.round(np.arange(1, 11) / 10, 2)
LEGIT = np.round(np.arange(1, 11) / 20, 2)
Y = np.r_[np.ones(10, bool), np.zeros(10, bool)]
S = np.r_[SCAM, LEGIT]


def test_threshold_meets_recall_with_fewest_alarms():
    # 90% of 10 scams = 9, so the 9th highest scam score (0.2) is the cut
    assert pick_threshold(Y, S, 0.9) == pytest.approx(0.2)
    op = at_threshold(Y, S, 0.2)
    assert op.recall == 0.9
    # genuine scores >= 0.2: 0.20, 0.25, ..., 0.50 = 7 of 10
    assert op.false_alarms_per_1000 == 700.0
    assert op.precision == round(9 / 16, 4)


def test_full_recall_uses_lowest_scam_score():
    assert pick_threshold(Y, S, 1.0) == pytest.approx(0.1)


def test_recall_target_is_rounded_up_not_down():
    # 0.95 of 10 scams cannot be 9.5: needs all 10
    assert at_threshold(Y, S, pick_threshold(Y, S, 0.95)).recall == 1.0


def test_threshold_comes_from_validation_not_test():
    val_y = np.array([1, 1, 0, 0], bool)
    val_s = np.array([0.9, 0.8, 0.1, 0.2])
    test_y = np.array([1, 1, 0, 0], bool)
    test_s = np.array([0.6, 0.5, 0.55, 0.1])
    out = evaluate_at_recall(val_y, val_s, test_y, test_s, target_recall=1.0)
    assert out["from_validation"]["threshold"] == pytest.approx(0.8)
    assert out["from_validation"]["recall"] == 0.0  # the threshold did not transfer
    assert out["oracle_not_for_reporting"]["recall"] == 1.0


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        pick_threshold(np.zeros(3, bool), np.ones(3), 0.9)
    with pytest.raises(ValueError):
        pick_threshold(Y, S, 0.0)
    with pytest.raises(ValueError):
        Bands(warn=0.8, flag=0.5)


def test_bands_partition_messages():
    bands = fit_bands(Y, S, warn_recall=1.0, flag_recall=0.5)
    assert bands.warn == pytest.approx(0.1) and bands.flag == pytest.approx(0.6)
    assert [bands.label(x) for x in (0.05, 0.3, 0.9)] == ["allow", "warn", "flag"]
    summary = band_summary(Y, S, bands)
    assert sum(summary[b]["scams"] + summary[b]["genuine"] for b in ("allow", "warn", "flag")) == 20
    assert summary["flag"]["genuine"] == 0


@pytest.fixture
def corpus():
    return pd.DataFrame(
        {
            "text": ["WIN a prize now", "see you at 6", "Your parcel fee", "Meeting moved", "Hi mum"],
            "domain": ["sms", "sms", "sms", "email", "sms"],
            "label": ["spam", "legit", "phishing", "legit", "phishing"],
            "split": ["train", "train", "test", "test", "val"],
        }
    )


def test_lock_catches_a_leak(tmp_path, corpus):
    path = tmp_path / "locked.json"
    m = write_manifest(corpus, path)
    assert m["n_messages"] == 2
    assert_unlocked(corpus[corpus.split == "train"], "training", path)
    # same message with different case and spacing is still the same message
    leaked = pd.DataFrame({"text": ["your  PARCEL fee!"], "domain": ["sms"]})
    with pytest.raises(LockedTestLeak):
        assert_unlocked(leaked, "threshold selection", path)


def test_same_text_other_channel_is_not_a_leak(tmp_path, corpus):
    path = tmp_path / "locked.json"
    write_manifest(corpus, path)
    assert_unlocked(pd.DataFrame({"text": ["Meeting moved"], "domain": ["sms"]}), "training", path)


def test_verify_detects_a_changed_split(tmp_path, corpus):
    path = tmp_path / "locked.json"
    write_manifest(corpus, path)
    verify(corpus, path)
    moved = corpus.assign(split=["train", "train", "train", "test", "val"])
    with pytest.raises(ValueError, match="test split changed"):
        verify(moved, path)


def test_edited_manifest_is_rejected(tmp_path, corpus):
    path = tmp_path / "locked.json"
    write_manifest(corpus, path)
    text = path.read_text().replace('"keys": [', '"keys": ["0000",', 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="edited"):
        load_keys(path)
