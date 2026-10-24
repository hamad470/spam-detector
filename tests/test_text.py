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


def test_prepare_adds_the_channel_prefix():
    assert prepare("hello", "sms") == "[sms] hello"
    assert prepare("Hello", "email") == "[email] hello"


def test_email_scrub_removes_corpus_fingerprints():
    raw = "Subject: [Spambayes] Re: price , only 15 $ ! mail jose@monkey.org\nhttp://xent.com/mailman/listinfo/fork"
    assert prepare(raw, "email") == "[email] subject: re: price, only xxmoney ! mail xxemail"


def test_email_addresses_survive_leetspeak_undoing():
    assert "xxemail" in prepare("contact supp0rt@paypa1.com now", "sms")


def test_enron_packaging_is_undone():
    from spam_detector.datasets import detokenize_enron

    assert detokenize_enron("visit http : / / www . x . com , it ' s high - income") == (
        "visit http://www.x.com, it's high-income"
    )


def test_years_are_masked():
    assert "xxyear" in prepare("your 2024 tax refund", "sms")


def test_llm_comparison_scores_refusals_as_wrong():
    from spam_detector.llm_compare import metrics

    m = metrics(["legit", "spam", "phishing", "legit"], ["legit", "phishing", "phishing", "error"])
    assert m["accuracy"] == 0.5 and m["errors"] == 1
    assert m["scams_caught"] == 1.0


def test_llm_prompt_lists_the_three_labels():
    from spam_detector.llm_compare import SCHEMA, system_prompt

    assert SCHEMA["properties"]["label"]["enum"] == ["legit", "spam", "phishing"]
    prompt = system_prompt("email", [("Win a prize now", "spam")])
    assert "email" in prompt and "Label: spam" in prompt
