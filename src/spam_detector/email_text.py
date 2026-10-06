"""Turn raw RFC 822 emails into the text a model should see.

Only the subject and the start of the body are kept. Headers are dropped on
purpose: routing headers, recipient addresses and mailer names identify which
*corpus* an email came from, not whether it's a scam.
"""

import contextlib
import email
import email.header
import email.policy
import html
import re
from email.message import Message

MAX_BODY_CHARS = 3000

_TAG = re.compile(r"<[^>]+>")
_STYLE = re.compile(r"<(style|script|head)[^>]*>.*?</\1>", re.S | re.I)
_BREAK = re.compile(r"<\s*(br|/p|/div|/tr|/li|/h\d)[^>]*>", re.I)
_QUOTED = re.compile(r"^\s*>.*$", re.M)
_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def html_to_text(raw: str) -> str:
    text = _STYLE.sub(" ", raw)
    text = _BREAK.sub("\n", text)
    text = html.unescape(_TAG.sub(" ", text))
    return text


def _decode(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "latin-1"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:  # made-up charset names are common in spam
        return payload.decode("latin-1", errors="replace")


def body_text(msg: Message) -> str:
    """Prefer text/plain; fall back to stripped HTML. Attachments are ignored."""
    plain, rich = [], []
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart" or part.get_filename():
            continue
        kind = part.get_content_type()
        if kind == "text/plain":
            plain.append(_decode(part))
        elif kind == "text/html":
            rich.append(html_to_text(_decode(part)))
    return "\n".join(plain) if any(p.strip() for p in plain) else "\n".join(rich)


def tidy(text: str) -> str:
    text = _QUOTED.sub("", text)
    text = _SPACES.sub(" ", text)
    return _BLANK_LINES.sub("\n", text).strip()


def subject_and_body(msg: Message) -> tuple[str, str]:
    subject = str(msg.get("Subject", "") or "")
    with contextlib.suppress(Exception):  # malformed encoded-words are common in spam
        subject = str(email.header.make_header(email.header.decode_header(subject)))
    return tidy(subject.replace("\n", " ")), tidy(body_text(msg))[:MAX_BODY_CHARS]


def parse_bytes(raw: bytes) -> Message:
    return email.message_from_bytes(raw, policy=email.policy.compat32)


def as_text(subject: str, body: str) -> str:
    return f"Subject: {subject}\n{body}".strip()
