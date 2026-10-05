"""FastAPI service: JSON API under /api, interactive docs at /docs, web UI at /."""

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spam_detector import __version__
from spam_detector.predict import DEFAULT_MODEL_DIR, SpamDetector

STATIC_DIR = Path(__file__).parent / "static"
MAX_CHARS = 5000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = SpamDetector(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
    yield


app = FastAPI(
    title="SMS Spam Detector",
    version=__version__,
    description="TF-IDF + chi-squared feature selection + Bernoulli Naive Bayes, "
    "trained on the UCI SMS Spam Collection. Returns a spam probability and the words behind it.",
    lifespan=lifespan,
)


class Evidence(BaseModel):
    word: str
    weight: float = Field(description="> 0 pushes towards spam, < 0 towards ham")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS, examples=["WINNER!! Claim your free prize now"])


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class PredictResponse(BaseModel):
    label: str
    spam_probability: float
    evidence: list[Evidence]


@app.post("/api/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    return asdict(request.app.state.detector.predict(body.text))


@app.post("/api/predict/batch", response_model=list[PredictResponse])
def predict_batch(body: BatchRequest, request: Request):
    texts = [t[:MAX_CHARS] for t in body.texts]
    return [asdict(p) for p in request.app.state.detector.predict_batch(texts)]


@app.get("/api/model")
def model_card(request: Request):
    return request.app.state.detector.card


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
