// Browser port of src/spam_detector/{text,decision,inference}.py.
// Same normaliser, same email clean-up, same tokenizer, same int8 ONNX model, same decision rule.
// web/test/parity.test.mjs checks every step against outputs from the Python code.

import * as ort from "onnxruntime-web";
import { Tokenizer } from "@huggingface/tokenizers";

export const LABELS = ["legit", "spam", "phishing"];

// ---- text.py: de-obfuscation ------------------------------------------------

const CONFUSABLES = {
  "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
  "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
  "С": "C", "Т": "T", "Х": "X", "ο": "o", "α": "a", "ν": "v",
};
const LEET = { 0: "o", 1: "i", 3: "e", 4: "a", 5: "s", 7: "t", "@": "a", $: "s" };
const SPACED_OUT = /\b[A-Za-z]([.\-_* ])[A-Za-z](?:\1[A-Za-z])+\b(?![.\-_*][A-Za-z])/g;
const ZERO_WIDTH = /[​-‏⁠﻿]/g;
const EMAIL_TOKEN = /[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_-]+\.[\p{L}\p{N}_.-]+/u;
const EMAIL_TOKEN_G = new RegExp(EMAIL_TOKEN.source, "gu");

function deleet(token) {
  if (EMAIL_TOKEN.test(token)) return token; // '@' is leetspeak for 'a', but not inside an address
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

// ---- text.py: entity tags ---------------------------------------------------

const TAGS = [
  [/[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_-]+(\.[\p{L}\p{N}_-]+)+/gu, " xxemail "],
  [/(https?:\/\/|www\.)\S+|\b\S+\.(com|co\.uk|net|org|ly|info|biz|io|xyz|top)\b/gi, " xxurl "],
  [/\+?\d[\d\s-]{7,}\d/g, " xxphone "],
  [/[£$€]\s?\d[\d,.]*|\d[\d,.]*\s?([£$€]|pounds\b|gbp\b|usd\b|dollars\b)/gi, " xxmoney "],
  [/\b(19[89]\d|20[0-3]\d)\b/g, " xxyear "],
];

export function tagEntities(text) {
  for (const [pattern, tag] of TAGS) text = text.replace(pattern, tag);
  return text;
}

// ---- text.py: email clean-up ------------------------------------------------

const SPACE_BEFORE = /\s+([,.;:!?%)\]}'])/g;
const SPACE_AFTER = /([(\[{$£€])\s+/g;
const LIST_TAG = /^(subject:\s*)((re|fwd?):\s*)*\[[^\]]{2,30}\]\s*/gim;
const LIST_FOOTER = /^.*(mailman\/listinfo|mailing list|list-unsubscribe|_{20,}|-{20,}).*$/gim;
const IDENTITY = /\b(jose|monkey|nazario|enron|kaminski|vince|ect|hou)\b/gi;

export function scrubEmail(text) {
  text = text.replace(EMAIL_TOKEN_G, " xxemail ");
  text = text.replace(LIST_TAG, "$1").replace(LIST_FOOTER, " ");
  text = text.toLowerCase();
  text = text.replace(SPACE_BEFORE, "$1").replace(SPACE_AFTER, "$1");
  return text.replace(IDENTITY, " ");
}

/** The exact string the model sees. Must match text.prepare() in Python. */
export function prepare(text, domain = "sms") {
  text = normalise(text);
  if (domain === "email") text = scrubEmail(text);
  return `[${domain}] ` + tagEntities(text).split(/\s+/).filter(Boolean).join(" ");
}

// ---- decision.py ----------------------------------------------------------

/** Scam first (P(spam) + P(phishing) >= threshold), then which kind. */
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

  encode(prepared, domain) {
    let ids = this.tokenizer.encode(prepared).ids;
    const max = this.config.max_len[domain];
    if (ids.length > max) ids = [...ids.slice(0, max - 1), this.eosId]; // keep </s>, like tokenizers' truncation
    return ids;
  }

  async #probs(prepared, domain) {
    const encoded = prepared.map((t) => this.encode(t, domain));
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

  async predictProba(texts, domain = "sms") {
    const rows = [];
    for (let i = 0; i < texts.length; i += 16) {
      rows.push(...(await this.#probs(texts.slice(i, i + 16).map((t) => prepare(t, domain)), domain)));
    }
    return rows;
  }

  async classify(text, { domain = "sms", explain = true, topK = 6 } = {}) {
    const [p] = await this.predictProba([text], domain);
    const scam = p[1] + p[2];
    const result = {
      label: LABELS[decide(p)],
      probabilities: Object.fromEntries(LABELS.map((l, i) => [l, p[i]])),
      scamProbability: scam,
      evidence: [],
    };
    if (explain) result.evidence = await this.#occlusion(text, domain, scam, topK);
    return result;
  }

  // Drop each word, re-score every variant, report the change in scam log-odds.
  // Emails: only the first 60 words, which is where the hook usually is.
  async #occlusion(text, domain, base, topK) {
    const spans = [...text.matchAll(/\S+/g)].slice(0, 60).map((m) => [m.index, m.index + m[0].length]);
    if (spans.length < 2) return [];
    const variants = spans.map(([a, b]) => prepare(text.slice(0, a) + text.slice(b), domain));
    const scores = [];
    for (let i = 0; i < variants.length; i += 16) {
      scores.push(...(await this.#probs(variants.slice(i, i + 16), domain)).map((p) => p[1] + p[2]));
    }
    const baseLo = logOdds(base);
    return spans
      .map(([a, b], i) => ({ word: text.slice(a, b), start: a, end: b, impact: baseLo - logOdds(scores[i]) }))
      .filter((e) => Math.abs(e.impact) >= 0.05)
      .sort((x, y) => Math.abs(y.impact) - Math.abs(x.impact))
      .slice(0, topK);
  }
}
