"""Reference outputs for the browser port.

    python -m spam_detector.web_fixtures

The web demo re-implements prepare(), tokenisation, the decision rule and the
occlusion explanations in JavaScript. This writes what the Python versions
produce for a few hundred messages, and web/test/parity.test.mjs checks the
JavaScript against it.
"""

import json
import random

import pandas as pd

from .attacks import attack_all
from .decision import decide
from .inference import ScamClassifier
from .paths import LABELS, MODELS, PROCESSED, ROOT
from .text import prepare

EDGE_CASES = [
    "ok",
    "   lots   of   spaces   ",
    "Call +44 7911 123456 or 0800-505-060 now!!",
    "Pay £1,250.00 or $40 or 300 pounds today",
    "visit www.example.co.uk or https://bit.ly/3xYz or prize.info",
    "c.l.a.i.m your F R E E p-r-i-z-e",
    "U.R.G.E.N.T Y*o*u*r account",
    "fr​ee pr‌ize",
    "Ｆｕｌｌｗｉｄｔｈ ＴＥＸＴ ５００",
    "Cаll nоw tо clаim",  # Cyrillic а and о
    "B1G W1NN3R 2day!!! txt 80086",
    "Émile a gagné 500€ ! Répondez OUI",
    "🎉🎉 You WON 🎁 claim at prize-🎁.com",
    "word " * 150,  # longer than the 96-token limit
]


def main() -> None:
    df = pd.read_parquet(PROCESSED / "corpus.parquet")
    test = df[(df.split == "test") & df.label.isin(LABELS)]
    rng = random.Random(0)
    sample = pd.concat(g.sample(min(len(g), 60), random_state=0) for _, g in test.groupby("label"))
    texts = list(sample.text)
    scams = list(sample[sample.label != "ham"].text)
    for attack in ["leetspeak", "homoglyph", "spacing", "typo", "mixed"]:
        texts += attack_all(rng.sample(scams, 12), attack, 0.5, seed=1)
    texts += EDGE_CASES

    clf = ScamClassifier(MODELS / "onnx")
    probs = clf.predict_proba(texts)
    cases = []
    for text, p in zip(texts, probs, strict=True):
        prepared = prepare(text)
        cases.append(
            {
                "text": text,
                "prepared": prepared,
                "ids": clf.tokenizer.encode(prepared).ids,
                "probs": [round(float(x), 6) for x in p],
                "label": LABELS[int(decide(p[None])[0])],
            }
        )
    # A few full explanations, to check the occlusion port end to end
    explained = [
        {
            "text": t,
            "evidence": [{"word": e["word"], "impact": e["impact"]} for e in clf.classify(t).evidence],
        }
        for t in texts[:5] + scams[:5]
    ]
    out = ROOT / "web" / "test" / "fixtures.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"cases": cases, "explained": explained}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {len(cases)} cases and {len(explained)} explanations to {out}")


if __name__ == "__main__":
    main()
