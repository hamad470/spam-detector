"""The locked final test set.

    python -m spam_detector.lock --write     # once, right after build_dataset (v4 protocol)
    python -m spam_detector.lock --verify    # any time: the corpus still matches the manifest

The test split is sealed by writing a manifest of hashed message keys to
reports/locked_test.json, which is committed. Hashes, not text, so the manifest says
nothing about the messages and can be public.

Anything that fits, tunes or designs (training, early stopping, temperature, thresholds,
bands, Gemini prompts and few-shot examples) must call `assert_unlocked(frame, purpose)`
on the rows it uses. It raises if a single locked message is among them.

Only `spam_detector.final_test` reads the locked rows, once per candidate model, and
logs that it did.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from .paths import PROCESSED, REPORTS
from .text import dedup_key

MANIFEST = REPORTS / "locked_test.json"
LOCKED_SPLITS = ("test",)


class LockedTestLeak(RuntimeError):
    """Raised when locked test messages reach a fitting or tuning step."""


def message_key(text: str, domain: str) -> str:
    """Stable id for a message: its channel plus its de-duplication key, hashed."""
    return hashlib.sha256(f"{domain}\x1f{dedup_key(text)}".encode()).hexdigest()


def keys_of(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [message_key(t, d) for t, d in zip(frame.text, frame.domain, strict=True)], index=frame.index
    )


def _digest(keys) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def write_manifest(corpus: pd.DataFrame, path=MANIFEST) -> dict:
    locked = corpus[corpus.split.isin(LOCKED_SPLITS)]
    keys = sorted(set(keys_of(locked)))
    manifest = {
        "splits": list(LOCKED_SPLITS),
        "n_messages": len(keys),
        "by_channel_label": {
            f"{d}/{lab}": int(n) for (d, lab), n in locked.groupby(["domain", "label"]).size().items()
        },
        "digest": _digest(keys),
        "keys": keys,
    }
    path.write_text(json.dumps(manifest, indent=1))
    return manifest


def load_keys(path=MANIFEST) -> frozenset[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m spam_detector.lock --write` first")
    manifest = json.loads(path.read_text())
    keys = manifest["keys"]
    if _digest(keys) != manifest["digest"]:
        raise ValueError(f"{path} has been edited: digest does not match its keys")
    return frozenset(keys)


def assert_unlocked(frame: pd.DataFrame, purpose: str, path=MANIFEST) -> None:
    """Raise LockedTestLeak if any row of `frame` is a locked test message."""
    locked = load_keys(path)
    hits = keys_of(frame).isin(locked)
    if hits.any():
        raise LockedTestLeak(
            f"{int(hits.sum())} locked test message(s) reached '{purpose}'. "
            "Only spam_detector.final_test may read the locked split."
        )


def verify(corpus: pd.DataFrame, path=MANIFEST) -> None:
    """The rebuilt corpus's test split must be exactly the sealed one."""
    locked = load_keys(path)
    now = frozenset(keys_of(corpus[corpus.split.isin(LOCKED_SPLITS)]))
    if now != locked:
        raise ValueError(
            f"test split changed: {len(now - locked)} new, {len(locked - now)} missing. "
            "The corpus was rebuilt differently; do not re-seal without logging why."
        )
    assert_unlocked(corpus[~corpus.split.isin(LOCKED_SPLITS)], "non-test splits", path)


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    args = p.parse_args()
    corpus = pd.read_parquet(PROCESSED / "corpus.parquet")
    if args.write:
        if MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} exists: the test set is sealed. Re-seal only with a logged reason.")
        m = write_manifest(corpus)
        print(f"sealed {m['n_messages']} messages, digest {m['digest'][:12]}")
    else:
        verify(corpus)
        print("locked test matches the manifest; no locked message in other splits")


if __name__ == "__main__":
    main()
