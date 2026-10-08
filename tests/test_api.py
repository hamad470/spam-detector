"""API and model behaviour. Needs the exported model in models/onnx (CI downloads it from the Hub)."""

import pytest
from fastapi.testclient import TestClient

from spam_detector.paths import MODELS

pytestmark = pytest.mark.skipif(not (MODELS / "onnx" / "model.onnx").exists(), reason="model not downloaded")

SMS_SCAM = "Royal Mail: your parcel is on hold due to an unpaid £1.99 fee. Pay at royalmail-redelivery-uk.com"
SMS_LEGIT = "Running 10 mins late, grab us a table and I'll be there soon"
EMAIL_PHISH = (
    "Subject: Action required: your password expires today\n\nDear user, your mailbox password expires in "
    "24 hours. Confirm your current password at https://m365-account-verify.top/login or your account "
    "will be suspended.\n\nIT Support"
)
EMAIL_LEGIT = (
    "Subject: Notes from Tuesday's meeting\n\nHi all, thanks for joining. Priya will share the updated "
    "timeline by Friday and I'll book a room for the client review next week.\n\nBest, Tom"
)


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def classify(client, text, kind="sms", **kw):
    r = client.post("/api/classify", json={"text": text, "type": kind, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


@pytest.mark.parametrize(
    "text, kind, scam",
    [
        (SMS_SCAM, "sms", True),
        (SMS_LEGIT, "sms", False),
        (EMAIL_PHISH, "email", True),
        (EMAIL_LEGIT, "email", False),
    ],
)
def test_obvious_cases(client, text, kind, scam):
    body = classify(client, text, kind)
    assert (body["scam_probability"] > 0.5) is scam
    assert (body["label"] != "legit") is scam
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_evidence_spans_point_into_the_text(client):
    body = classify(client, SMS_SCAM)
    assert body["evidence"]
    for e in body["evidence"]:
        assert SMS_SCAM[e["start"] : e["end"]] == e["word"]


def test_obfuscation_does_not_hide_a_scam(client):
    body = classify(client, "Y0ur acc0unt is l0cked. Ver1fy y0ur det4ils at secure-l0gin.net")
    assert body["scam_probability"] > 0.5


def test_validation(client):
    assert client.post("/api/classify", json={"text": ""}).status_code == 422
    assert client.post("/api/classify", json={"text": "a" * 6001}).status_code == 422
    assert client.post("/api/classify", json={"text": "hi", "type": "fax"}).status_code == 422


def test_batch_skips_explanations(client):
    out = client.post("/api/classify/batch", json={"texts": [SMS_SCAM, SMS_LEGIT]}).json()
    assert [o["evidence"] for o in out] == [[], []]


def test_model_card_and_page(client):
    assert "headline_model" in client.get("/api/model").json()
    assert "Scam Detector" in client.get("/").text
    assert client.get("/scam.js").status_code == 200
