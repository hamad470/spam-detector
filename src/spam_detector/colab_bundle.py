"""Package the code and the processed corpus for training on Google Colab.

    python -m spam_detector.colab_bundle      # writes artifacts/colab_bundle.zip

Upload the zip in notebooks/train_on_colab.ipynb. The corpus checksum printed here
should match the one the notebook prints, so both sides train on identical data.
"""

import hashlib
import zipfile

from .paths import ARTIFACTS, PROCESSED, ROOT


def main() -> None:
    corpus = PROCESSED / "corpus.parquet"
    out = ARTIFACTS / "colab_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in (ROOT / "src").rglob("*.py"):
            zf.write(path, path.relative_to(ROOT))
        for name in ("pyproject.toml", "README.md"):
            zf.write(ROOT / name, name)
        zf.write(corpus, "data/processed/corpus.parquet")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("corpus sha256:", hashlib.sha256(corpus.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
