from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"  # trained models (large ones are gitignored)
MODELS = ROOT / "models"  # small, committed serving artefacts
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

LABELS = ["ham", "spam", "smishing"]
