"""Multi-key handling in llm_compare, with fake clients: no network, no real quota used."""

import json
from types import SimpleNamespace

import pytest
from google.genai import errors

from spam_detector import llm_compare
from spam_detector.llm_compare import GeminiClassifier, QuotaExhausted, load_env_keys, mask

QUOTA = errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})


class FakeClient:
    """Answers `label` until `budget` requests are used, then refuses with 429 like the real API."""

    def __init__(self, label="spam", budget=10**9):
        self.label, self.budget, self.calls = label, budget, 0
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **_):
        self.calls += 1
        if self.calls > self.budget:
            raise QUOTA
        return SimpleNamespace(text=json.dumps({"label": self.label}))


def classifier(clients):
    by_key = {f"key-{i}": c for i, c in enumerate(clients)}
    return GeminiClassifier("fake-model", list(by_key), rpm=10**6, make_client=by_key.__getitem__, backoff=0)


def test_uses_the_first_key_while_it_has_quota():
    first, second = FakeClient("phishing"), FakeClient("legit")
    clf = classifier([first, second])
    assert [clf.classify("hi", "sys")[0] for _ in range(3)] == ["phishing"] * 3
    assert second.calls == 0


def test_switches_to_the_next_key_when_quota_runs_out():
    first, second = FakeClient("spam", budget=2), FakeClient("legit")
    clf = classifier([first, second])
    labels = [clf.classify("hi", "sys")[0] for _ in range(4)]
    assert labels == ["spam", "spam", "legit", "legit"]
    assert clf.active == 1


def test_stops_cleanly_when_every_key_is_out_of_quota():
    clf = classifier([FakeClient(budget=0), FakeClient(budget=0)])
    with pytest.raises(QuotaExhausted):
        clf.classify("hi", "sys")


def test_a_label_outside_the_three_is_an_error():
    assert classifier([FakeClient("maybe")]).classify("hi", "sys")[0] == "error"


def test_keys_load_from_env_file_in_order(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(llm_compare, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEYS= aaaa1111 , bbbb2222 \n", encoding="utf-8")
    assert load_env_keys() == ["aaaa1111", "bbbb2222"]
    (tmp_path / ".env").write_text("GEMINI_API_KEY=cccc3333\n", encoding="utf-8")
    assert load_env_keys() == ["cccc3333"]


def test_keys_are_never_printed_in_full():
    assert mask("AIzaSyExampleKey1234") == "...1234"
