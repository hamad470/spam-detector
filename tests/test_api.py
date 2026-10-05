import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_predict(client):
    r = client.post("/api/predict", json={"text": "WINNER! Claim your free prize now, text WIN to 80086"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "spam"
    assert 0 <= body["spam_probability"] <= 1
    assert {"word", "weight"} <= body["evidence"][0].keys()


def test_predict_rejects_empty_and_oversized_text(client):
    assert client.post("/api/predict", json={"text": ""}).status_code == 422
    assert client.post("/api/predict", json={"text": "a" * 5001}).status_code == 422


def test_batch(client):
    r = client.post("/api/predict/batch", json={"texts": ["free prize claim now", "see you at 7"]})
    assert [p["label"] for p in r.json()] == ["spam", "ham"]


def test_model_card_and_ui(client):
    assert client.get("/api/model").json()["model"] == "bernoulli_nb"
    assert "SMS Spam Detector" in client.get("/").text
