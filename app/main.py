"""FastAPI service: JSON API under /api, Swagger at /docs, and the same page as the live demo at /."""

import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spam_detector import __version__
from spam_detector.inference import ScamClassifier
from spam_detector.paths import MODELS

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
MODEL_DIR = Path(os.getenv("MODEL_DIR", MODELS / "onnx"))
MODEL_REPO = os.getenv("MODEL_REPO", "hamadurrehman62/scam-detector-distilroberta")
MAX_CHARS = 6000  # emails are cut to ~256 tokens anyway


def _ensure_model() -> None:
    if (MODEL_DIR / "model.onnx").exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_model()
    app.state.clf = ScamClassifier(MODEL_DIR)
    card = MODEL_DIR / "metrics.json"
    app.state.card = json.loads(card.read_text()) if card.exists() else {}
    yield


app = FastAPI(
    title="SMS & Email Scam Detector",
    version=__version__,
    description="Fine-tuned DistilRoBERTa (int8 ONNX) that separates legitimate messages, marketing spam "
    "and phishing in SMS and email, with per-word explanations.",
    lifespan=lifespan,
)


class Evidence(BaseModel):
    word: str
    start: int
    end: int
    impact: float = Field(description="Drop in scam log-odds when the word is removed (> 0: pointed to scam)")


class ClassifyRequest(BaseModel):
    text: str = Field(
        min_length=1, max_length=MAX_CHARS, examples=["Parcel on hold. Pay £1.99 at royalmail-fee.com"]
    )
    type: Literal["sms", "email"] = "sms"
    explain: bool = True


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    type: Literal["sms", "email"] = "sms"


class ClassifyResponse(BaseModel):
    label: str = Field(description="legit, spam or phishing")
    probabilities: dict[str, float]
    scam_probability: float = Field(description="P(spam) + P(phishing)")
    evidence: list[Evidence]


@app.post("/api/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest, request: Request):
    return asdict(request.app.state.clf.classify(body.text, body.type, explain=body.explain))


@app.post("/api/classify/batch", response_model=list[ClassifyResponse])
def classify_batch(body: BatchRequest, request: Request):
    clf = request.app.state.clf
    return [asdict(clf.classify(t[:MAX_CHARS], body.type, explain=False)) for t in body.texts]


@app.get("/api/model")
def model_card(request: Request):
    return request.app.state.card


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


# The page runs the model in the browser, so it works the same here as on the static Space
app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
