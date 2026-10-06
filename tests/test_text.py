import random

import pytest

from spam_detector.attacks import ATTACKS, attack_all
from spam_detector.dedup import near_duplicate_groups
from spam_detector.text import dedup_key, fix_encoding, normalise, prepare, tag_entities

MESSAGE = "URGENT your account is blocked, claim your prize now"


def test_fix_encoding_repairs_mojibake():
    assert fix_encoding("Win Â£500 now") == "Win £500 now"


def test_dedup_key_ignores_case_spacing_and_encoding():
    assert dedup_key("Win  Â£500 NOW!") == dedup_key("win £500 now")


@pytest.mark.parametrize("attack", ["leetspeak", "homoglyph"])
def test_normalise_undoes_character_attacks(attack):
    attacked = attack_all([MESSAGE], attack, rate=1.0, seed=3)[0]
    assert attacked != MESSAGE
    assert normalise(attacked).lower() == MESSAGE.lower()


def test_normalise_joins_spaced_out_words():
    assert normalise("c.l.a.i.m your F R E E p-r-i-z-e") == "claim your FREE prize"
    assert normalise("U.R.G.E.N.T Y*o*u*r account") == "URGENT Your account"


def test_normalise_keeps_phone_numbers_and_prices():
    assert normalise("call 09061790121 for £500") == "call 09061790121 for £500"


def test_attacks_are_deterministic_for_a_seed():
    for name in [*ATTACKS, "mixed"]:
        assert attack_all([MESSAGE], name, 0.5, seed=7) == attack_all([MESSAGE], name, 0.5, seed=7)


def test_rate_zero_is_a_no_op():
    rng = random.Random(0)
    for fn in ATTACKS.values():
        assert fn(MESSAGE, 0.0, rng) == MESSAGE


def test_tag_entities():
    tagged = tag_entities("Pay £1.99 at www.royalmail-fee.com or call +44 7911 123456")
    assert {"xxmoney", "xxurl", "xxphone"} <= set(tagged.split())


def test_prepare_combines_both():
    assert "xxurl" in prepare("V1sit www.prize.com").split()


def test_near_duplicate_groups_catch_templated_campaigns():
    texts = [
        "URGENT! You have won a £900 prize. Call 09061701444 to claim. Valid 12hrs",
        "URGENT! You have won a £900 prize. Call 09061701461 to claim. Valid 12hrs",
        "are we still on for dinner tonight?",
    ]
    groups = near_duplicate_groups(texts)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_decide_flags_scam_even_when_ham_wins_argmax():
    import numpy as np

    from spam_detector.decision import decide

    proba = np.array([[0.41, 0.30, 0.29], [0.90, 0.05, 0.05], [0.20, 0.10, 0.70]])
    assert decide(proba).tolist() == [1, 0, 2]
