// Does the browser port behave like the Python service?
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

test("prepare() matches Python on every fixture", () => {
  for (const c of fixtures.cases) assert.equal(prepare(c.text), c.prepared, JSON.stringify(c.text));
});

test("decision rule matches Python", () => {
  for (const c of fixtures.cases) assert.equal(LABELS[decide(c.probs)], c.label);
});

test("token ids match Python (including truncation)", { skip: !haveModel && "model not downloaded" }, () => {
  for (const c of fixtures.cases) assert.deepEqual(clf.encode(c.prepared), c.ids, JSON.stringify(c.text));
});

// WebAssembly and native ONNX Runtime round int8 matmuls differently, so probabilities
// aren't bit-identical. Labels must be; typical drift must stay tiny.
test("labels match Python and probabilities stay close", { skip: !haveModel && "model not downloaded" }, async () => {
  const probs = await clf.predictProba(fixtures.cases.map((c) => c.text));
  const diffs = fixtures.cases.map((c, i) => {
    assert.equal(LABELS[decide(probs[i])], c.label, JSON.stringify(c.text));
    return Math.max(...probs[i].map((p, j) => Math.abs(p - c.probs[j])));
  });
  diffs.sort((a, b) => a - b);
  const median = diffs[Math.floor(diffs.length / 2)];
  assert.ok(median < 1e-3, `median probability difference ${median}`);
  assert.ok(diffs.at(-1) < 0.15, `max probability difference ${diffs.at(-1)}`);
});

test("strong explanation words match Python", { skip: !haveModel && "model not downloaded" }, async () => {
  for (const e of fixtures.explained) {
    const words = new Set((await clf.classify(e.text)).evidence.map((x) => x.word));
    for (const x of e.evidence.filter((x) => Math.abs(x.impact) >= 0.5)) {
      assert.ok(words.has(x.word), `"${x.word}" missing for ${JSON.stringify(e.text)}`);
    }
  }
});
