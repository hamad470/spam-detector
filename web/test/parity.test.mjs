// Does the browser port behave like the Python code?
// Fixtures come from `python -m spam_detector.web_fixtures`; the model from models/onnx (or the Hub in CI).

import { test, before } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { LABELS, ScamClassifier, decide, prepare } from "../scam.js";

const fixtures = JSON.parse(await readFile(new URL("./fixtures.json", import.meta.url), "utf8"));
const modelDir = new URL("../../models/onnx/", import.meta.url);
const haveModel = existsSync(new URL("model.onnx", modelDir));
let clf;

before(async () => {
  if (!haveModel) return;
  clf = await ScamClassifier.create({
    modelBytes: new Uint8Array(await readFile(new URL("model.onnx", modelDir))),
    tokenizerJson: JSON.parse(await readFile(new URL("tokenizer.json", modelDir), "utf8")),
    config: JSON.parse(await readFile(new URL("config.json", modelDir), "utf8")),
  });
});

const label = (c) => `${c.domain}: ${JSON.stringify(c.text.slice(0, 80))}`;

test("prepare() matches Python on every fixture, SMS and email", () => {
  for (const c of fixtures.cases) assert.equal(prepare(c.text, c.domain), c.prepared, label(c));
});

test("decision rule matches Python", () => {
  for (const c of fixtures.cases) assert.equal(LABELS[decide(c.probs)], c.label);
});

test("token ids match Python, including per-channel truncation", { skip: !haveModel && "model not downloaded" }, () => {
  for (const c of fixtures.cases) assert.deepEqual(clf.encode(c.prepared, c.domain), c.ids, label(c));
});

// WebAssembly and native ONNX Runtime round int8 matrix products differently, so probabilities
// aren't bit-identical. Typical drift must stay tiny, and labels may only differ on messages
// that sit right at the decision boundary.
test("labels match Python and probabilities stay close", { skip: !haveModel && "model not downloaded" }, async () => {
  const diffs = [];
  for (const domain of ["sms", "email"]) {
    const cases = fixtures.cases.filter((c) => c.domain === domain);
    const probs = await clf.predictProba(cases.map((c) => c.text), domain);
    cases.forEach((c, i) => {
      const gap = Math.max(...probs[i].map((p, j) => Math.abs(p - c.probs[j])));
      const borderline = Math.abs(1 - c.probs[0] - 0.5) < 0.1 || Math.abs(c.probs[1] - c.probs[2]) < 0.1;
      if (!borderline) assert.equal(LABELS[decide(probs[i])], c.label, label(c));
      diffs.push(gap);
    });
  }
  diffs.sort((a, b) => a - b);
  const median = diffs[Math.floor(diffs.length / 2)];
  assert.ok(median < 2e-3, `median probability difference ${median}`);
});

// Rounding differences reorder near-ties (especially far from p = 0.5, where log-odds are
// sensitive), so a strong Python word must be in the browser's top 12 and point the same way.
test("strong explanation words match Python", { skip: !haveModel && "model not downloaded" }, async () => {
  for (const e of fixtures.explained) {
    const r = await clf.classify(e.text, { domain: e.domain, topK: 12 });
    const js = new Map(r.evidence.map((x) => [x.word, x.impact]));
    for (const x of e.evidence.filter((x) => Math.abs(x.impact) >= 0.5)) {
      assert.ok(js.has(x.word), `"${x.word}" missing for ${label(e)}`);
      assert.equal(Math.sign(js.get(x.word)), Math.sign(x.impact), `"${x.word}" flips for ${label(e)}`);
    }
  }
});
