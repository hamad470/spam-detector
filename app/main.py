"""FastAPI service: JSON API under /api, Swagger at /docs, web UI at /."""

import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spam_detector import __version__
from spam_detector.inference import ScamClassifier
from spam_detector.paths import MODELS

STATIC_DIR = Path(__file__).parent / "static"
MODEL_DIR = Path(os.getenv("MODEL_DIR", MODELS / "onnx"))
MODEL_REPO = os.getenv("MODEL_REPO", "hamad470/sms-scam-distilroberta")
MAX_CHARS = 1000  # an SMS is 160 chars; long-form text is out of distribution anyway


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
    title="SMS Scam Detector",
    version=__version__,
    description="Fine-tuned DistilRoBERTa (int8 ONNX) that separates ham, marketing spam and smishing, "
    "with per-word explanations. Trained on de-duplicated UCI 2011 + Mendeley 2022 SMS data.",
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
    explain: bool = True


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class ClassifyResponse(BaseModel):
    label: str = Field(description="ham, spam or smishing")
    probabilities: dict[str, float]
    scam_probability: float = Field(description="P(spam) + P(smishing)")
    evidence: list[Evidence]


@app.post("/api/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest, request: Request):
    return asdict(request.app.state.clf.classify(body.text, explain=body.explain))


@app.post("/api/classify/batch", response_model=list[ClassifyResponse])
def classify_batch(body: BatchRequest, request: Request):
    clf = request.app.state.clf
    return [asdict(clf.classify(t[:MAX_CHARS], explain=False)) for t in body.texts]


@app.get("/api/model")
def model_card(request: Request):
    return request.app.state.card


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
