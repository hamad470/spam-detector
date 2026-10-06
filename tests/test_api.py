"""API and model behaviour. Needs the exported model in models/onnx (CI downloads it from the Hub)."""

import pytest
from fastapi.testclient import TestClient

from spam_detector.paths import MODELS

pytestmark = pytest.mark.skipif(
    not (MODELS / "onnx" / "model.onnx").exists(), reason="model not exported/downloaded"
)

SMISHING = "Royal Mail: your parcel is on hold due to an unpaid £1.99 fee. Pay at royalmail-redelivery-uk.com"
HAM = "Running 10 mins late, grab us a table and I'll be there soon"


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_classify_scam_and_ham(client):
    scam = client.post("/api/classify", json={"text": SMISHING}).json()
    ham = client.post("/api/classify", json={"text": HAM}).json()
    assert scam["label"] in {"spam", "smishing"} and scam["scam_probability"] > 0.5
    assert ham["label"] == "ham" and ham["scam_probability"] < 0.5
    assert abs(sum(scam["probabilities"].values()) - 1) < 1e-3


def test_evidence_spans_point_into_the_text(client):
    body = client.post("/api/classify", json={"text": SMISHING}).json()
    assert body["evidence"]
    for e in body["evidence"]:
        assert SMISHING[e["start"] : e["end"]] == e["word"]


def test_obfuscation_does_not_hide_a_scam(client):
    text = "Y0ur acc0unt is l0cked. Ver1fy y0ur det4ils at secure-l0gin.net"
    body = client.post("/api/classify", json={"text": text}).json()
    assert body["scam_probability"] > 0.5


def test_validation(client):
    assert client.post("/api/classify", json={"text": ""}).status_code == 422
    assert client.post("/api/classify", json={"text": "a" * 1001}).status_code == 422


def test_batch_skips_explanations(client):
    out = client.post("/api/classify/batch", json={"texts": [SMISHING, HAM]}).json()
    assert [o["evidence"] for o in out] == [[], []]


def test_model_card_and_ui(client):
    assert "final_model" in client.get("/api/model").json()
    assert "SMS Scam Detector" in client.get("/").text
