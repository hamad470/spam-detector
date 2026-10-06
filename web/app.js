import * as ort from "onnxruntime-web";
import { ScamClassifier } from "./scam.js";

const MODEL_REPO = "https://huggingface.co/hamadurrehman62/sms-scam-distilroberta/resolve/main/";
const CACHE = "sms-scam-model-v1";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const CLASSES = {
  ham: { title: "Looks legitimate", cls: "ham" },
  spam: { title: "Marketing spam", cls: "spam" },
  smishing: { title: "Smishing: likely a scam", cls: "smishing" },
};
const MODEL_NAMES = {
  distilroberta_aug: "DistilRoBERTa + attack augmentation",
  distilroberta: "DistilRoBERTa",
  minilm_lr: "MiniLM embeddings + LR",
  tfidf_lr: "TF-IDF word+char + LR",
  nb_original: "Naive Bayes (original notebook)",
};

let clf;

// The model is 82 MB, so keep it in Cache Storage rather than trusting the HTTP cache.
async function fetchModel(onProgress) {
  const url = MODEL_REPO + "model.onnx";
  let cache;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null; // private mode or blocked storage: just download
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 82_522_752;
  const reader = res.body.getReader();
  const bytes = new Uint8Array(total);
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes.set(value, received);
    received += value.length;
    onProgress(received / total);
  }
  const model = bytes.subarray(0, received);
  try {
    await cache?.put(url, new Response(model));
  } catch {
    /* quota exceeded: fine, it'll download again next visit */
  }
  return model;
}

async function load() {
  const progress = (f) => {
    $("load-bar").style.width = pct(f, 0);
    $("load-pct").textContent = pct(f, 0);
  };
  const [config, tokenizerJson, card, modelBytes] = await Promise.all([
    fetch(MODEL_REPO + "config.json").then((r) => r.json()),
    fetch(MODEL_REPO + "tokenizer.json").then((r) => r.json()),
    fetch(MODEL_REPO + "metrics.json").then((r) => r.json()),
    fetchModel(progress),
  ]);
  $("load-text").textContent = "Starting the model…";
  clf = await ScamClassifier.create({ modelBytes, tokenizerJson, config });
  renderCard(card);
  $("loader").hidden = true;
  $("go").disabled = false;

  const shared = new URLSearchParams(location.search).get("text");
  if (shared) {
    $("msg").value = shared;
    check();
  }
}

async function check() {
  const text = $("msg").value.trim();
  if (!text) return $("msg").focus();
  if (!clf) return;
  $("go").disabled = true;
  try {
    const started = performance.now();
    const r = await clf.classify(text);
    render(text, r, performance.now() - started);
  } catch (err) {
    alert("Something went wrong: " + err.message);
  } finally {
    $("go").disabled = false;
  }
}

function render(text, r, ms) {
  const c = CLASSES[r.label];
  $("verdict").textContent = c.title;
  $("verdict").className = "verdict " + c.cls;
  $("headline").textContent = `scam probability ${pct(r.scamProbability)}`;

  const probs = Object.entries(r.probabilities);
  $("stack").innerHTML = probs
    .map(([k, v]) => `<span class="seg ${k}" style="flex-grow:${Math.max(v, 0.002)}" title="${k} ${pct(v)}"></span>`)
    .join("");
  $("stack").setAttribute("aria-label", probs.map(([k, v]) => `${k} ${pct(v)}`).join(", "));
  $("legend").innerHTML = probs.map(([k, v]) => `<span><i class="sw ${k}"></i>${k} <b>${pct(v)}</b></span>`).join("");

  // Highlight by character span so a repeated word is marked where it actually occurs
  const spans = [...r.evidence].sort((a, b) => a.start - b.start);
  let html = "";
  let pos = 0;
  for (const e of spans) {
    html += esc(text.slice(pos, e.start));
    html += `<mark class="${e.impact > 0 ? "s" : "h"}" title="impact ${e.impact.toFixed(2)}">${esc(text.slice(e.start, e.end))}</mark>`;
    pos = e.end;
  }
  $("highlighted").innerHTML = html + esc(text.slice(pos));

  const max = Math.max(...r.evidence.map((e) => Math.abs(e.impact)), 0.1);
  $("evidence").innerHTML = r.evidence.length
    ? r.evidence
        .map((e) => `<li><span class="w">${esc(e.word)}</span>
          <span class="b ${e.impact > 0 ? "s" : "h"}" style="width:${(Math.abs(e.impact) / max) * 100}%"></span>
          <span class="n">${e.impact > 0 ? "+" : ""}${e.impact.toFixed(2)}</span></li>`)
        .join("")
    : '<li class="muted">No single word moves the decision much. The verdict comes from the message as a whole.</li>';
  $("timing").textContent = `Computed on your device in ${Math.round(ms)} ms.`;
  $("result").hidden = false;
}

function renderCard(c) {
  const o = c.onnx;
  $("model-desc").textContent =
    `${MODEL_NAMES[c.final_model]}, quantised to int8 ONNX (${o.onnx_size_mb} MB). ` +
    `Tested on ${c.test_messages.toLocaleString()} messages from spam campaigns it never saw during training.`;
  $("tiles").innerHTML = [
    ["Macro-F1 (3 classes)", pct(o.test_macro_f1)],
    ["Scams caught", pct(o.test_scam_recall)],
    ["Smishing recall", pct(o.test_smishing_recall)],
    ["Real texts flagged", pct(o.test_ham_false_positive_rate, 2)],
    ["Robust @30% obfuscation", pct(c.models[c.final_model].robustness.mixed["0.3"])],
  ]
    .map(([k, v]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`)
    .join("");
  $("cmp").innerHTML =
    "<tr><th>Model</th><th>Macro-F1</th><th>Scam F1</th><th>Robustness</th><th>Size</th></tr>" +
    Object.entries(c.models)
      .sort((a, b) => b[1].macro_f1 - a[1].macro_f1)
      .map(([n, s]) => `<tr class="${n === c.final_model ? "best" : ""}"><td>${MODEL_NAMES[n] || n}</td>
        <td>${pct(s.macro_f1)}</td><td>${pct(s.binary.f1)}</td><td>${pct(s.robustness.mixed["0.3"])}</td><td>${s.size_mb} MB</td></tr>`)
      .join("");
}

$("go").addEventListener("click", check);
$("msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) check();
});
$("examples").addEventListener("click", (e) => {
  if (!e.target.dataset.text) return;
  $("msg").value = e.target.dataset.text;
  check();
});

load().catch((err) => {
  $("load-text").textContent = "Couldn't load the model: " + err.message;
  $("load-bar").style.background = "var(--spam)";
});
