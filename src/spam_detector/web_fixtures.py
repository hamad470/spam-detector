"""Reference outputs for the browser port.

    python -m spam_detector.web_fixtures

The web demo re-implements prepare(), tokenisation, the decision rule and the
occlusion explanations in JavaScript. This writes what the Python versions
produce for a few hundred messages from both channels, and web/test/parity.test.mjs
checks the JavaScript against it.
"""

import json

import pandas as pd

from .attacks import attack_all
from .decision import decide
from .inference import ScamClassifier
from .paths import LABELS, MODELS, PROCESSED, ROOT
from .text import prepare

SMS_EDGE_CASES = [
    "ok",
    "   lots   of   spaces   ",
    "Call +44 7911 123456 or 0800-505-060 now!!",
    "Pay £1,250.00 or $40 or 300 pounds today, 15$ only",
    "visit www.example.co.uk or https://bit.ly/3xYz or prize.info or app.io",
    "c.l.a.i.m your F R E E p-r-i-z-e",
    "U.R.G.E.N.T Y*o*u*r account",
    "fr​ee pr‌ize",
    "Ｆｕｌｌｗｉｄｔｈ ＴＥＸＴ ５００",
    "Cаll nоw tо clаim",  # Cyrillic а and о
    "B1G W1NN3R 2day!!! txt 80086 in 2024",
    "mail supp0rt@paypa1.com or s.mith+x@mail.co.uk",
    "Émile a gagné 500€ ! Répondez OUI",
    "🎉🎉 You WON 🎁 claim at prize-🎁.com",
    "word " * 150,
]
EMAIL_EDGE_CASES = [
    "Subject: [Spambayes] Re: price , only 15 $ ! email jose@monkey.org\nhttp://xent.com/mailman/listinfo/fork",
    "Subject: Fwd: [ILUG] meeting\nHi all,\n> quoted line\nSee you at Enron, Vince\n" + "-" * 30 + "\nfooter",
    "Subject: Your Account Has Been LOCKED\nDear Customer ( valued ) , click https://login.top/x now !",
    "Subject: invoice 2023\n" + "lorem ipsum dolor sit amet " * 80,  # longer than the 256-token limit
]


def main() -> None:
    df = pd.read_parquet(PROCESSED / "corpus.parquet")
    test = df[(df.split == "test") & df.label.isin(LABELS)]
    sample = pd.concat(
        g.sample(min(len(g), 30), random_state=0) for _, g in test.groupby(["domain", "label"])
    )
    cases = list(zip(sample.text, sample.domain, strict=True))
    scams = sample[sample.label != "legit"]
    for attack in ["leetspeak", "homoglyph", "spacing", "typo", "mixed"]:
        picked = scams.sample(10, random_state=len(attack))
        cases += list(zip(attack_all(picked.text, attack, 0.5, seed=1), picked.domain, strict=True))
    cases += [(t, "sms") for t in SMS_EDGE_CASES] + [(t, "email") for t in EMAIL_EDGE_CASES]

    clf = ScamClassifier(MODELS / "onnx")
    probs = clf.predict_proba([t for t, _ in cases], [d for _, d in cases])
    out_cases = []
    for (text, domain), p in zip(cases, probs, strict=True):
        prepared = prepare(text, domain)
        clf.tokenizer.enable_truncation(clf.max_len[domain])
        out_cases.append(
            {
                "text": text,
                "domain": domain,
                "prepared": prepared,
                "ids": clf.tokenizer.encode(prepared).ids,
                "probs": [round(float(x), 6) for x in p],
                "label": LABELS[int(decide(p[None])[0])],
            }
        )
    explain = [c for c in cases if c[1] == "sms"][:6] + [c for c in cases if c[1] == "email"][:3]
    explained = [
        {
            "text": t,
            "domain": d,
            "evidence": [{"word": e["word"], "impact": e["impact"]} for e in clf.classify(t, d).evidence],
        }
        for t, d in explain
    ]
    out = ROOT / "web" / "test" / "fixtures.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"cases": out_cases, "explained": explained}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"wrote {len(out_cases)} cases and {len(explained)} explanations to {out}")


if __name__ == "__main__":
    main()
