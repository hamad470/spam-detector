"""Loading the UCI SMS Spam Collection."""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "spam.csv"


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Return a de-duplicated frame with columns ``text`` and ``label`` (1 = spam)."""
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["target", "text"]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = (df["target"] == "spam").astype(int)
    return df[["text", "label"]]
