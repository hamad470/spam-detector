// Browser port of src/spam_detector/{text,decision,inference}.py.
// Same normaliser, same tokenizer, same int8 ONNX model, same decision rule.
// web/test/parity.test.mjs checks every step against outputs from the Python code.

import * as ort from "onnxruntime-web";
import { Tokenizer } from "@huggingface/tokenizers";

export const LABELS = ["ham", "spam", "smishing"];

// ---- text.py --------------------------------------------------------------

const CONFUSABLES = {
  "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
  "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
  "С": "C", "Т": "T", "Х": "X", "ο": "o", "α": "a", "ν": "v",
};
const LEET = { 0: "o", 1: "i", 3: "e", 4: "a", 5: "s", 7: "t", "@": "a", $: "s" };
const SPACED_OUT = /\b[A-Za-z]([.\-_* ])[A-Za-z](?:\1[A-Za-z])+\b(?![.\-_*][A-Za-z])/g;
const ZERO_WIDTH = /[​-‏⁠﻿]/g;
const TAGS = [
  [/(https?:\/\/|www\.)\S+|\b\S+\.(com|co\.uk|net|ly|info|biz)\b/gi, " xxurl "],
  [/\+?\d[\d\s-]{7,}\d/g, " xxphone "],
  [/[£$€]\s?\d[\d,.]*|\d[\d,.]*\s?(pounds|gbp|usd)\b/gi, " xxmoney "],
];

function deleet(token) {
  const chars = [...token];
  const hasAlpha = chars.some((c) => /\p{L}/u.test(c));
  const mostlyDigits = chars.filter((c) => /\p{Nd}/u.test(c)).length > chars.length / 2;
  return hasAlpha && !mostlyDigits ? chars.map((c) => LEET[c] ?? c).join("") : token;
}

export function normalise(text) {
  text = text.normalize("NFKC").replace(ZERO_WIDTH, "");
  text = [...text].map((c) => CONFUSABLES[c] ?? c).join("");
  text = text.replace(SPACED_OUT, (m) => m.replace(/[.\-_* ]/g, ""));
  return text.replace(/\S+/g, deleet);
}

export function tagEntities(text) {
  for (const [pattern, tag] of TAGS) text = text.replace(pattern, tag);
  return text;
}

export const prepare = (text) => tagEntities(normalise(text));

// ---- decision.py ----------------------------------------------------------

/** Scam first (P(spam) + P(smishing) >= threshold), then which kind. */
export function decide(p, threshold = 0.5) {
  if (1 - p[0] < threshold) return 0;
  return p[2] > p[1] ? 2 : 1;
}

// ---- inference.py ---------------------------------------------------------

const logOdds = (p, eps = 1e-6) => {
  p = Math.min(Math.max(p, eps), 1 - eps);
  return Math.log(p / (1 - p));
};

export class ScamClassifier {
  /** Build from raw files so the same code runs in the browser (fetched) and Node (read from disk). */
  static async create({ modelBytes, tokenizerJson, config, sessionOptions = {} }) {
    const self = new ScamClassifier();
    self.config = config;
    self.tokenizer = new Tokenizer(tokenizerJson, {});
    self.padId = tokenizerJson.model.vocab["<pad>"];
    self.eosId = tokenizerJson.model.vocab["</s>"];
    self.session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"], ...sessionOptions });
    return self;
  }

  encode(prepared) {
    let ids = this.tokenizer.encode(prepared).ids;
    const max = this.config.max_len;
    if (ids.length > max) ids = [...ids.slice(0, max - 1), this.eosId]; // keep </s>, like tokenizers' truncation
    return ids;
  }

  async #probs(prepared) {
    const encoded = prepared.map((t) => this.encode(t));
    const width = Math.max(...encoded.map((ids) => ids.length));
    const ids = new BigInt64Array(encoded.length * width).fill(BigInt(this.padId));
    const mask = new BigInt64Array(encoded.length * width);
    encoded.forEach((row, i) =>
      row.forEach((id, j) => {
        ids[i * width + j] = BigInt(id);
        mask[i * width + j] = 1n;
      }),
    );
    const dims = [encoded.length, width];
    const out = await this.session.run({
      input_ids: new ort.Tensor("int64", ids, dims),
      attention_mask: new ort.Tensor("int64", mask, dims),
    });
    const z = out.logits.data;
    const k = this.config.labels.length;
    const rows = [];
    for (let i = 0; i < encoded.length; i++) {
      const row = Array.from(z.slice(i * k, (i + 1) * k), (v) => v / this.config.temperature);
      const top = Math.max(...row);
      const e = row.map((v) => Math.exp(v - top));
      const sum = e.reduce((a, b) => a + b, 0);
      rows.push(e.map((v) => v / sum));
    }
    return rows;
  }

  async predictProba(texts) {
    const rows = [];
    for (let i = 0; i < texts.length; i += 32) rows.push(...(await this.#probs(texts.slice(i, i + 32).map(prepare))));
    return rows;
  }

  async classify(text, { explain = true, topK = 6 } = {}) {
    const [p] = await this.predictProba([text]);
    const scam = p[1] + p[2];
    const result = {
      label: LABELS[decide(p)],
      probabilities: Object.fromEntries(LABELS.map((l, i) => [l, p[i]])),
      scamProbability: scam,
      evidence: [],
    };
    if (explain) result.evidence = await this.#occlusion(text, scam, topK);
    return result;
  }

  // Drop each word, re-score every variant in one batch, report the change in scam log-odds
  async #occlusion(text, base, topK) {
    const spans = [...text.matchAll(/\S+/g)].slice(0, 60).map((m) => [m.index, m.index + m[0].length]);
    if (spans.length < 2) return [];
    const variants = spans.map(([a, b]) => prepare(text.slice(0, a) + text.slice(b)));
    const scores = (await this.#probs(variants)).map((p) => p[1] + p[2]);
    const baseLo = logOdds(base);
    return spans
      .map(([a, b], i) => ({ word: text.slice(a, b), start: a, end: b, impact: baseLo - logOdds(scores[i]) }))
      .filter((e) => Math.abs(e.impact) >= 0.05)
      .sort((x, y) => Math.abs(y.impact) - Math.abs(x.impact))
      .slice(0, topK);
  }
}
