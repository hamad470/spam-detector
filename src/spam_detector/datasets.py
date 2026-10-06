"""Raw dataset download and loading.

    python -m spam_detector.datasets      # fetch anything missing into data/raw/

Every loader returns a frame with: text, label (source's own label), domain (sms/email),
source, year. Mapping source labels onto legit/spam/phishing happens in build_dataset.
"""

import glob
import io
import mailbox
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .email_text import as_text, parse_bytes, subject_and_body
from .paths import RAW
from .text import fix_encoding

NAZARIO = "https://monkey.org/~jose/phishing/"
# phishing3.mbox (2007) is left out: Windows Defender quarantines it on download because it carries
# live malicious payloads, and I'd rather lose 1 of 15 files than switch protection off.
NAZARIO_FILES = ["phishing0.mbox", "phishing1.mbox", "phishing2.mbox", "20051114.mbox"] + [
    f"phishing-{y}" for y in range(2015, 2026)
]
SPAMASSASSIN = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_FILES = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20050311_spam_2.tar.bz2",
]
HF_ENRON = "https://huggingface.co/api/datasets/SetFit/enron_spam/parquet/default/"

SOURCES = {
    # UCI SMS Spam Collection (Almeida & Gomez Hidalgo, 2011), CC BY 4.0
    "uci_sms_spam.csv": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    # SMS Phishing Dataset (Mishra & Soni, 2022), Mendeley Data, DOI 10.17632/f45bkkt8pr.1, CC BY 4.0
    "mendeley/Dataset_5971.csv": "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
    "edb361de-918d-469f-9106-e84823830665/file_downloaded",
    # ExAIS_SMS (Abayomi-Alli et al., 2022), collected with participants' consent; no licence file
    "exais/exais.zip": "https://github.com/AbayomiAlli/SMS-Spam-Dataset/raw/main/ExAIS_SMS%20Spam%20Dataset.zip",
    # Enron-Spam (Metsis et al., 2006) as packaged by SetFit on the Hugging Face Hub
    "enron_train.parquet": HF_ENRON + "train/0.parquet",
    "enron_test.parquet": HF_ENRON + "test/0.parquet",
    # Nazario phishing corpus, hand-screened phishing from 2004 to 2025, CC BY 4.0
    **{f"nazario/{f}": NAZARIO + f for f in NAZARIO_FILES},
    # SpamAssassin public corpus (2002-2005)
    **{f"spamassassin/{f}": SPAMASSASSIN + f for f in SPAMASSASSIN_FILES},
}


def _fetch(url: str) -> bytes:
    # Mendeley returns 403 to urllib's default user agent
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (spam-detector data script)"})
    return urllib.request.urlopen(req, timeout=120).read()


def download(force: bool = False) -> None:
    for name, url in SOURCES.items():
        dest = RAW / name
        if dest.exists() and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}")
        payload = _fetch(url)
        if name == "uci_sms_spam.csv":  # the UCI zip ships a headerless TSV
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                df = pd.read_csv(zf.open("SMSSpamCollection"), sep="\t", header=None, names=["v1", "v2"])
            payload = df.to_csv(index=False).encode("latin-1", errors="replace")
        elif name.startswith("mendeley/") and payload[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                payload = zf.read(next(n for n in zf.namelist() if n.endswith(".csv")))
        dest.write_bytes(payload)
        if name.endswith(".zip"):
            zipfile.ZipFile(dest).extractall(dest.parent)
        elif name.endswith(".tar.bz2"):
            with tarfile.open(dest) as tf:
                tf.extractall(dest.parent, filter="data")


# ---- SMS -----------------------------------------------------------------


def load_uci() -> pd.DataFrame:
    df = pd.read_csv(RAW / "uci_sms_spam.csv", encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    return df.assign(text=df.text.map(fix_encoding), domain="sms", source="uci_2011", year=2011)


def load_mendeley() -> pd.DataFrame:
    df = pd.read_csv(RAW / "mendeley" / "Dataset_5971.csv", encoding="latin-1")
    # Labels arrive as ham / spam / Spam / smishing / Smishing
    return pd.DataFrame(
        {
            "label": df.LABEL.str.lower(),
            "text": df.TEXT.map(fix_encoding),
            "domain": "sms",
            "source": "mendeley_2022",
            "year": 2022,
        }
    )


def load_exais() -> pd.DataFrame:
    """The per-user CSVs have no header and unquoted commas inside messages, so parse by hand.

    Columns: type, direction, sender, sender, timestamp, ?, HAM/SPAM, message...
    """
    rows = []
    for path in sorted(glob.glob(str(RAW / "exais" / "ExAIS_SMS Spam Dataset" / "*.csv"))):
        for line in Path(path).read_bytes().decode("utf-8", errors="replace").splitlines():
            parts = line.split(",")
            if len(parts) < 8 or parts[6].strip().upper() not in ("HAM", "SPAM"):
                continue  # 5 lines with a broken label field
            text = ",".join(parts[7:]).rstrip(",").strip().strip('"')
            # Trailing columns sometimes repeat the label in lowercase
            text = re.sub(r",*\s*(spam|ham)\s*$", "", text, flags=re.I).strip()
            year = re.search(r"(20\d\d)", parts[4])
            rows.append(
                {
                    "label": parts[6].strip().lower(),
                    "text": fix_encoding(text),
                    "direction": parts[1].strip(),
                    "year": int(year.group(1)) if year else 2014,
                }
            )
    return pd.DataFrame(rows).assign(domain="sms", source="exais_2014")


# ---- Email ---------------------------------------------------------------


_ENRON_FIXES = [
    (re.compile(r"\b(https?) : / / "), r"\1://"),
    (re.compile(r"(\w) ([.@/:]) (?=\w)"), r"\1\2"),  # www . x . com, jm @ x . com, 12 / 13
    (re.compile(r"(\w) ([-']) (?=\w)"), r"\1\2"),  # high - income, it ' s
    (re.compile(r" ([,.;:!?%)\]])"), r"\1"),
    (re.compile(r"([(\[$£]) "), r"\1"),
    (re.compile(r"(?:_ ){3,}_?"), "_____"),
]


def detokenize_enron(text: str) -> str:
    """Undo the SetFit packaging, which puts spaces around every punctuation mark.

    Left in, that formatting alone tells a model "this is Enron, so it's not phishing".
    """
    for pattern, repl in _ENRON_FIXES:
        text = pattern.sub(repl, text)
    return text


def load_enron() -> pd.DataFrame:
    """All 33,716 Enron-Spam emails. Lower-cased by the packager; spacing is undone here."""
    df = pd.concat(
        [pd.read_parquet(RAW / f"enron_{s}.parquet") for s in ("train", "test")], ignore_index=True
    )
    packaged = [
        as_text(s, b[:3000]) for s, b in zip(df.subject.fillna(""), df.message.fillna(""), strict=True)
    ]
    text = [
        as_text(detokenize_enron(s), detokenize_enron(b[:3000]))
        for s, b in zip(df.subject.fillna(""), df.message.fillna(""), strict=True)
    ]
    return pd.DataFrame(
        {
            "label": df.label.map({0: "ham", 1: "spam"}),
            "text": [fix_encoding(t) for t in text],
            "text_as_packaged": packaged,  # kept only for the shortcut audit
            "domain": "email",
            "source": "enron",
            "year": pd.to_datetime(df.date, errors="coerce").dt.year.fillna(2004).astype(int),
        }
    )


def _email_rows(raw_messages, label: str, source: str, default_year: int):
    for raw in raw_messages:
        msg = parse_bytes(raw)
        subject, body = subject_and_body(msg)
        year = re.search(r"\b(19|20)\d\d\b", str(msg.get("Date", "")))
        yield {
            "label": label,
            "text": fix_encoding(as_text(subject, body)),
            "domain": "email",
            "source": source,
            "year": int(year.group()) if year else default_year,
        }


def load_spamassassin() -> pd.DataFrame:
    rows = []
    for folder, label in [("easy_ham", "ham"), ("easy_ham_2", "ham"), ("hard_ham", "hard_ham"),
                          ("spam", "spam"), ("spam_2", "spam")]:  # fmt: skip
        files = [p for p in sorted((RAW / "spamassassin" / folder).iterdir()) if p.name != "cmds"]
        rows += _email_rows((p.read_bytes() for p in files), label, "spamassassin", 2002)
    return pd.DataFrame(rows)


def load_nazario() -> pd.DataFrame:
    rows = []
    for name in NAZARIO_FILES:
        box = mailbox.mbox(RAW / "nazario" / name, create=False)
        year = int(name[-4:]) if name.startswith("phishing-") else 2005
        rows += [
            r | {"file": name} for r in _email_rows((m.as_bytes() for m in box), "phishing", "nazario", year)
        ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    download()
