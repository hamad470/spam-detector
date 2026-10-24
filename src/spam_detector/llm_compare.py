"""Fine-tuned model vs Gemini on the same held-out messages.

    # once: put your key in .env at the repo root (gitignored):  GEMINI_API_KEY=...
    # several keys, used in order when one runs out of quota:    GEMINI_API_KEYS=key1,key2
    python -m spam_detector.llm_compare                       # writes web/compare.json
    python -m spam_detector.llm_compare --model gemini-3.8-flash --rpm 10

The pool comes from the validation split: the deployed model was retrained on the test split, so
validation is the held-out data it has never been fitted on (it was only used to stop training and
fit the temperature). Gemini is forced to answer with exactly one label through structured output,
at temperature 0. Every answer is cached in artifacts/gemini_cache.jsonl, so a rerun after a quota
error or a disconnect only asks for what's missing.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .decision import decide
from .paths import ARTIFACTS, DOMAINS, LABELS, MODELS, PROCESSED, REPORTS, ROOT

SEED = 7
PER_CLASS = {"legit": 60, "spam": 45, "phishing": 45}  # per channel; 300 messages in total
DISPLAY_CHARS = 1200
CACHE = ARTIFACTS / "gemini_cache.jsonl"

# Handwritten messages that mimic what people actually receive today: genuine service texts the
# training data lacks, and scams with no link. Illustrative, not a benchmark.
PROBES = [
    ("sms", "legit", "Your verification code is 482913. Do not share it with anyone."),
    ("sms", "legit", "Reminder: your dentist appointment is tomorrow at 10:30. Reply C to cancel"),
    ("sms", "legit", "Your Uber driver is arriving in 3 minutes. Look for a white Toyota Prius"),
    ("sms", "legit", "Your parcel will be delivered today between 10:00 and 12:00. Track it in the app."),
    ("sms", "legit", "Your bank card ending 4421 was used for £23.50 at Tesco. If this wasn't you, call the number "
     "on the back of your card."),
    ("sms", "phishing", "Hi mum, I've dropped my phone in the toilet, this is my new number. Can you send me £200 "
     "for rent, I'll pay you back tomorrow"),
    ("sms", "phishing", "Hello, is this still your number? I have a job offer for you, part time from home, earn "
     "up to 800 a day. Reply for details"),
    ("sms", "phishing", "HMRC: You are eligible for a tax refund of £326.40. Claim now at hmrc-refund-gov.uk"),
    ("sms", "phishing", "Royal Mail: your parcel is on hold due to an unpaid £1.99 fee. Pay at royalmail-redelivery-uk.com"),
    ("email", "legit", "Subject: Your Amazon order #114-2235 has shipped\n\nHi, your package will arrive Tuesday. "
     "Track your order in Your Orders on amazon.co.uk."),
    ("email", "legit", "Subject: Re: Q3 budget\n\nHi Sam, attached is the revised Q3 budget. Can you check the travel "
     "line before Thursday? Thanks, Anna"),
    ("email", "phishing", "Subject: Urgent request\n\nAre you available? I need you to buy some gift cards for a "
     "client today, I'll reimburse you. Reply asap. Sent from my iPhone"),
]  # fmt: skip

DEFINITIONS = """You classify one message into exactly one of three labels.

legit: anything the recipient would want or reasonably expect: personal messages, work email,
genuine notifications, newsletters they signed up for, genuine service messages.
spam: unsolicited bulk marketing or promotions: offers, ads, prize draws or subscriptions that
try to sell something, without trying to steal money or credentials by deception.
phishing: a scam or fraud attempt: impersonating a bank, delivery company, employer or person;
asking for credentials, payment, personal details or a call/click under false pretences;
fake prizes or refunds designed to extract money or data.

The message is a {kind}. Answer with the label only."""

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": LABELS}},
    "required": ["label"],
}


def load_env_keys() -> list[str]:
    """API keys in priority order.

    GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY, from the environment or from a .env file at
    the repo root. With several keys, the next one is used only when the current one is out of quota
    (for example a primary and a backup key on a paid plan).
    """
    values = {k: os.environ.get(k, "") for k in ("GEMINI_API_KEYS", "GEMINI_API_KEY")}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in values and not values[name.strip()]:
                values[name.strip()] = value.strip().strip('"').strip("'")
    raw = values["GEMINI_API_KEYS"] or values["GEMINI_API_KEY"]
    return [k.strip() for k in raw.split(",") if k.strip()]


def mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) > 8 else "..."


def build_pool(corpus: pd.DataFrame) -> pd.DataFrame:
    val = corpus[(corpus.split == "val") & corpus.label.isin(LABELS)]
    parts = []
    for d in DOMAINS:
        for label, n in PER_CLASS.items():
            g = val[(val.domain == d) & (val.label == label)]
            parts.append(g.sample(min(n, len(g)), random_state=SEED))
    pool = pd.concat(parts)
    return pool.assign(id=[f"{d}-{i}" for d, i in zip(pool.domain, pool.index, strict=True)])


def few_shot_examples(corpus: pd.DataFrame, domain: str, per_label: int = 2) -> list[tuple[str, str]]:
    """Short, clear training-split examples. Never drawn from the pool's split."""
    train = corpus[(corpus.split == "train") & (corpus.domain == domain) & corpus.label.isin(LABELS)]
    train = train[train.text.str.len().between(60, 500 if domain == "sms" else 900)]
    rng = random.Random(SEED)
    examples = []
    for label in LABELS:
        rows = train[train.label == label]
        for i in rng.sample(range(len(rows)), per_label):
            examples.append((rows.text.iloc[i], label))
    rng.shuffle(examples)
    return examples


def system_prompt(domain: str, examples: list[tuple[str, str]] | None) -> str:
    prompt = DEFINITIONS.format(kind="text message (SMS)" if domain == "sms" else "email")
    if examples:
        shots = "\n\n".join(f"Message:\n{t}\nLabel: {y}" for t, y in examples)
        prompt += f"\n\nLabelled examples:\n\n{shots}"
    return prompt


class QuotaExhausted(RuntimeError):
    """The free tier's daily request limit is used up; answers so far are cached."""


class GeminiClassifier:
    def __init__(self, model: str, keys: list[str], rpm: int, make_client=None, backoff: float = 5.0):
        if make_client is None:
            from google import genai

            def make_client(key):
                return genai.Client(api_key=key)

        if not keys:
            raise ValueError("no API keys")
        self.keys = keys
        self.clients = [make_client(k) for k in keys]
        self.active = 0
        self.model = model
        self.min_gap = 60.0 / rpm
        self.backoff = backoff
        self._last = 0.0

    def classify(self, text: str, system: str) -> tuple[str, float]:
        """One label per message. Moves to the next key only when the current one is out of quota."""
        while True:
            try:
                return self._classify_with(self.clients[self.active], text, system)
            except QuotaExhausted:
                if self.active + 1 >= len(self.clients):
                    raise
                self.active += 1
                old, new = mask(self.keys[self.active - 1]), mask(self.keys[self.active])
                print(f"  key {old} is out of quota; switching to {new}")

    def _classify_with(self, client, text: str, system: str) -> tuple[str, float]:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=SCHEMA,
        )
        for attempt in range(6):
            wait = self._last + self.min_gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            started = time.perf_counter()
            try:
                r = client.models.generate_content(model=self.model, contents=text, config=config)
                label = json.loads(r.text)["label"]
                return (label if label in LABELS else "error"), time.perf_counter() - started
            except errors.APIError as e:
                if e.code in (429, 500, 503) and attempt < 5:
                    time.sleep(min(60, self.backoff * 2**attempt))  # rate limit or overload: back off
                    continue
                if e.code == 429:  # still refused after backing off: this key's quota is gone
                    raise QuotaExhausted(str(e)[:300]) from e
                print(f"  gemini error {e.code}: {str(e)[:120]}")
                return "error", time.perf_counter() - started
            except (json.JSONDecodeError, KeyError, TypeError):
                return "error", time.perf_counter() - started  # blocked or empty answer
        return "error", 0.0


def read_cache() -> dict:
    if not CACHE.exists():
        return {}
    rows = [json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["model"], r["variant"], r["id"]): r for r in rows}


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Errors and refusals count as wrong; they're reported separately as well."""
    y = np.array([LABELS.index(v) for v in y_true])
    p = np.array([LABELS.index(v) if v in LABELS else -1 for v in y_pred])
    present = sorted(set(y.tolist()))
    p_eval = np.where(p < 0, (y + 1) % 3, p)  # a refusal can't be right
    scam, flagged = y > 0, p_eval > 0

    def macro(a, b):
        return f1_score(a, b, average="macro", labels=present, zero_division=0)

    rng = np.random.default_rng(SEED)
    boot = [macro(y[s], p_eval[s]) for s in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    return {
        "n": len(y),
        "accuracy": round(float((y == p).mean()), 4),
        "macro_f1": round(float(macro(y, p_eval)), 4),
        "macro_f1_ci95": [
            round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4),
        ],
        "scams_caught": round(float(flagged[scam].mean()), 4),
        "legit_flagged": round(float(flagged[~scam].mean()), 4),
        "phishing_recall": round(float((p_eval[y == 2] == 2).mean()), 4),
        "errors": int((p < 0).sum()),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--rpm", type=int, default=10, help="requests per minute; keep under your quota")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pool messages (smoke test)")
    args = ap.parse_args(argv)

    keys = load_env_keys()
    if not keys:
        raise SystemExit(
            "No API key. Add GEMINI_API_KEY=... (or GEMINI_API_KEYS=k1,k2) to .env in the repo root."
        )
    print(f"{len(keys)} API key(s): " + ", ".join(mask(k) for k in keys))

    corpus = pd.read_parquet(PROCESSED / "corpus.parquet", columns=["text", "label", "domain", "split"])
    pool = build_pool(corpus)
    if args.limit:
        pool = pool.groupby("domain").head(args.limit // 2)
    print(f"pool: {len(pool)} validation messages {pool.groupby(['domain', 'label']).size().to_dict()}")

    # The deployed model (int8 ONNX), exactly what the website runs
    from .inference import ScamClassifier

    clf = ScamClassifier(MODELS / "onnx")
    proba = clf.predict_proba(pool.text, pool.domain)
    pool["model"] = np.array(LABELS)[decide(proba)]
    pool["model_scam_p"] = (1 - proba[:, 0]).round(4)
    probes = pd.DataFrame(PROBES, columns=["domain", "label", "text"])
    probes["id"] = [f"probe-{i}" for i in range(len(probes))]
    probes["model"] = np.array(LABELS)[decide(clf.predict_proba(probes.text, probes.domain))]
    if args.limit:
        probes = probes.head(0)

    variants = {
        "zero_shot": {d: system_prompt(d, None) for d in DOMAINS},
        "few_shot": {d: system_prompt(d, few_shot_examples(corpus, d)) for d in DOMAINS},
    }
    gemini = GeminiClassifier(args.model, keys, args.rpm)
    cache = read_cache()
    ARTIFACTS.mkdir(exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as out:
        for variant, prompts in variants.items():
            todo = [
                r
                for r in [*pool.itertuples(), *probes.itertuples()]
                if (args.model, variant, r.id) not in cache
            ]
            minutes = len(todo) / args.rpm
            print(f"{variant}: {len(pool) - len(todo)} cached, {len(todo)} to ask (~{minutes:.0f} min)")
            for i, r in enumerate(todo, 1):
                try:
                    label, secs = gemini.classify(r.text[:6000], prompts[r.domain])
                except QuotaExhausted as e:
                    left = sum((args.model, v, m) not in cache for v in variants for m in pool.id)
                    print(f"\nStopped: Gemini's daily quota is used up ({e}).")
                    raise SystemExit(f"{left} answers still needed. Rerun after the quota resets.") from e
                row = {
                    "model": args.model,
                    "variant": variant,
                    "id": r.id,
                    "label": label,
                    "secs": round(secs, 3),
                }
                cache[(args.model, variant, r.id)] = row
                out.write(json.dumps(row) + "\n")
                out.flush()
                if i % 25 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
    for variant in variants:
        probes[variant] = [cache[(args.model, variant, i)]["label"] for i in probes.id]
        pool[variant] = [cache[(args.model, variant, i)]["label"] for i in pool.id]
        pool[f"{variant}_secs"] = [cache[(args.model, variant, i)]["secs"] for i in pool.id]

    systems = {"model": "model", "gemini_zero_shot": "zero_shot", "gemini_few_shot": "few_shot"}
    summary = {
        name: {
            scope: metrics(part.label.tolist(), part[col].tolist())
            for scope, part in [("all", pool), *[(d, pool[pool.domain == d]) for d in DOMAINS]]
        }
        for name, col in systems.items()
    }
    for name in ("gemini_zero_shot", "gemini_few_shot"):
        summary[name]["median_seconds"] = round(float(pool[f"{systems[name]}_secs"].median()), 3)
    agree = (pool.zero_shot == pool.few_shot).mean()

    payload = {
        "gemini_model": args.model,
        "split": "validation",
        "pool_size": len(pool),
        "prompts": {v: {d: p for d, p in ps.items()} for v, ps in variants.items()},
        "summary": summary,
        "gemini_prompts_agree": round(float(agree), 4),
        "probes": [
            {
                "domain": r.domain,
                "label": r.label,
                "text": r.text,
                "model": r.model,
                "gemini_zero_shot": r.zero_shot,
                "gemini_few_shot": r.few_shot,
            }
            for r in probes.itertuples()
        ],  # fmt: skip
        "messages": [
            {
                "id": r.id,
                "domain": r.domain,
                "text": r.text[:DISPLAY_CHARS] + ("…" if len(r.text) > DISPLAY_CHARS else ""),
                "label": r.label,
                "model": r.model,
                "model_scam_p": r.model_scam_p,
                "gemini_zero_shot": r.zero_shot,
                "gemini_few_shot": r.few_shot,
            }
            for r in pool.itertuples()
        ],
    }
    if not args.limit:
        (ROOT / "web" / "compare.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "llm_comparison.json").write_text(
            json.dumps({k: v for k, v in payload.items() if k != "messages"}, indent=2), encoding="utf-8"
        )
    for name, s in summary.items():
        a = s["all"]
        caught, flagged = a["scams_caught"], a["legit_flagged"]
        print(f"{name:18} acc {a['accuracy']:.3f}  macro-F1 {a['macro_f1']:.3f}  scams caught {caught:.1%}")
        print(f"{'':18} legit flagged {flagged:.1%}  errors {a['errors']}")


if __name__ == "__main__":
    main()
