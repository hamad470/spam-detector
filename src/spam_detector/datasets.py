"""Raw dataset download and loading.

python -m spam_detector.datasets      # fetch anything missing into data/raw/
"""

import io
import urllib.request
import zipfile

import pandas as pd

from .paths import RAW
from .text import fix_encoding

SOURCES = {
    # UCI SMS Spam Collection (Almeida & Gomez Hidalgo, 2011), CC BY 4.0
    "uci_sms_spam.csv": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    # SMS Phishing Dataset (Mishra & Soni, 2022), Mendeley Data, DOI 10.17632/f45bkkt8pr.1, CC BY 4.0
    "mendeley/Dataset_5971.csv": "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
    "edb361de-918d-469f-9106-e84823830665/file_downloaded",
    # Enron-Spam (Metsis et al., 2006) as packaged by SetFit on the Hugging Face Hub
    "enron_test.parquet": "https://huggingface.co/api/datasets/SetFit/enron_spam/parquet/default/test/0.parquet",
}


def download(force: bool = False) -> None:
    for name, url in SOURCES.items():
        dest = RAW / name
        if dest.exists() and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}")
        # Mendeley returns 403 to urllib's default user agent
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (spam-detector data script)"})
        payload = urllib.request.urlopen(req, timeout=60).read()
        if url.endswith(".zip") or payload[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                member = next(n for n in zf.namelist() if n.endswith((".csv", "SMSSpamCollection")))
                payload = zf.read(member)
                if member.endswith("SMSSpamCollection"):  # the UCI zip ships a TSV without header
                    df = pd.read_csv(io.BytesIO(payload), sep="\t", header=None, names=["v1", "v2"])
                    payload = df.to_csv(index=False).encode("latin-1", errors="replace")
        dest.write_bytes(payload)


def load_uci() -> pd.DataFrame:
    df = pd.read_csv(RAW / "uci_sms_spam.csv", encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    return df.assign(text=df.text.map(fix_encoding), source="uci_2011")


def load_mendeley() -> pd.DataFrame:
    df = pd.read_csv(RAW / "mendeley" / "Dataset_5971.csv", encoding="latin-1")
    # Labels arrive as ham / spam / Spam / smishing / Smishing
    return pd.DataFrame(
        {"label": df.LABEL.str.lower(), "text": df.TEXT.map(fix_encoding), "source": "mendeley_2022"}
    )


def load_enron(max_chars: int = 600) -> pd.DataFrame:
    """Email test set for the out-of-domain check. Subject + start of body, roughly SMS-sized."""
    df = pd.read_parquet(RAW / "enron_test.parquet")
    text = (df.subject.fillna("") + "\n" + df.message.fillna("")).str.slice(0, max_chars)
    return pd.DataFrame(
        {"text": text.map(fix_encoding), "label": df.label.map({0: "ham", 1: "spam"}), "source": "enron"}
    )


if __name__ == "__main__":
    download()
