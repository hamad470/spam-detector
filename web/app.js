import * as ort from "onnxruntime-web";
import { ScamClassifier } from "./scam.js";

const HUB = "https://huggingface.co/hamadurrehman62/scam-detector-distilroberta/resolve/main/";
// Local development: serve the repo root and open /web/, and the page uses the freshly exported model
const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname) && location.pathname.startsWith("/web/");
const MODEL_REPO = LOCAL ? "../models/onnx/" : HUB;
const CACHE = LOCAL ? "scam-model-local" : "scam-model-v3";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x == null ? "n/a" : (x * 100).toFixed(d) + "%");
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const CLASSES = {
  legit: { title: "Looks legitimate", cls: "legit" },
  spam: { title: "Marketing spam", cls: "spam" },
  phishing: { title: "Phishing: likely a scam", cls: "phishing" },
};
const MODEL_NAMES = {
  distilroberta: "DistilRoBERTa (fine-tuned, this demo)",
  tfidf_lr: "TF-IDF word+char + logistic regression",
  nb_original: "Naive Bayes (my original notebook)",
};
const EXAMPLES = {
  sms: [
    ["Parcel fee", "Royal Mail: your parcel is on hold due to an unpaid £1.99 shipping fee. Pay now to avoid return: royalmail-redelivery-uk.com"],
    ["Bank alert", "HSBC: A new payee was added to your account. If this was NOT you, call our fraud team immediately on 07418 355612"],
    ["Obfuscated", "Y0ur acc0unt has been l0cked. Ver1fy y0ur det4ils at secure-l0gin.net within 24hrs"],
    ["Promo", "50% off all pizzas this weekend only! Order online with code WKND50. Txt STOP to opt out"],
    ["Friend", "Running 10 mins late, grab us a table and I'll be there soon"],
  ],
  email: [
    [
      "Password expiry",
      "Subject: Action required: your Microsoft 365 password expires today\n\nDear user,\n\nYour mailbox password will expire in 24 hours. To keep using your account without interruption, confirm your current password using the secure link below.\n\nhttps://m365-account-verify.top/login\n\nIf you do not act now your mailbox will be suspended and incoming messages deleted.\n\nIT Support Team",
    ],
    [
      "Invoice",
      "Subject: Outstanding invoice INV-20931 - payment overdue\n\nHello,\n\nPlease find your overdue invoice attached. Our records show £2,450.00 remains unpaid. To avoid late fees, settle the balance today via our secure payment portal: https://docs-invoice-pay.com/inv20931\n\nKind regards,\nAccounts Receivable",
    ],
    [
      "Newsletter",
      "Subject: Spring sale: 30% off everything this weekend\n\nHi there,\n\nOur spring sale starts Friday. Get 30% off all garden furniture and free delivery on orders over £50. Shop now before stocks run out.\n\nYou are receiving this email because you signed up at our store. Unsubscribe at any time.",
    ],
    [
      "Colleague",
      "Subject: Notes from Tuesday's project meeting\n\nHi all,\n\nThanks for joining on Tuesday. Main actions: Priya will share the updated timeline by Friday, and I'll book a room for the client review next week. Let me know if I've missed anything.\n\nBest,\nTom",
    ],
  ],
};

let clf;
let domain = "sms";

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
  const total = Number(res.headers.get("content-length")) || 82_500_000;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(received / total, 1));
  }
  const model = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    model.set(c, offset);
    offset += c.length;
  }
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
    fetch(MODEL_REPO + "metrics.json").then((r) => (r.ok ? r.json() : null)),
    fetchModel(progress),
  ]);
  $("load-text").textContent = "Starting the model…";
  clf = await ScamClassifier.create({ modelBytes, tokenizerJson, config });
  if (card) renderCard(card);
  else $("model-desc").textContent = "Model card not available.";
  $("loader").hidden = true;
  $("go").disabled = false;

  const params = new URLSearchParams(location.search);
  if (params.get("text")) {
    setDomain(params.get("type") === "email" ? "email" : "sms");
    $("msg").value = params.get("text");
    check();
  }
}

function setDomain(d) {
  domain = d;
  for (const b of document.querySelectorAll(".seg-btn")) {
    const on = b.dataset.domain === d;
    b.classList.toggle("active", on);
    b.setAttribute("aria-checked", String(on));
  }
  $("msg").placeholder =
    d === "sms" ? "e.g. Royal Mail: your parcel is on hold. Pay £1.99 at royalmail-redelivery.com" : "Paste the subject line and body of an email";
  $("msg").rows = d === "sms" ? 4 : 9;
  $("examples").innerHTML =
    '<span class="muted small">Try:</span>' +
    EXAMPLES[d].map(([name, text], i) => `<button class="chip" data-i="${i}" title="${esc(text.slice(0, 120))}">${name}</button>`).join("");
  $("result").hidden = true;
}

async function check() {
  const text = $("msg").value.trim();
  if (!text) return $("msg").focus();
  if (!clf) return;
  $("go").disabled = true;
  $("go").textContent = domain === "email" ? "Checking… (a few seconds)" : "Checking…";
  try {
    const started = performance.now();
    const r = await clf.classify(text, { domain });
    render(text, r, performance.now() - started);
  } catch (err) {
    alert("Something went wrong: " + err.message);
  } finally {
    $("go").disabled = false;
    $("go").textContent = "Check";
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
  const scope = domain === "email" ? " For emails, only the first 60 words are explained." : "";
  $("timing").textContent = `Computed on your device in ${Math.round(ms)} ms.${scope}`;
  $("result").hidden = false;
}

// spam_unlabelled has no 3-class label, so it isn't part of the headline test sets
const count = (sizes) => ["legit", "spam", "phishing"].reduce((n, k) => n + (sizes[k] || 0), 0);

function renderCard(c) {
  const m = c.models[c.headline_model];
  $("model-desc").textContent =
    `Fine-tuned DistilRoBERTa, quantised to int8 ONNX (${c.onnx.onnx_size_mb} MB). Scores are the mean of ${c.n_seeds} training runs, ` +
    `tested on spam campaigns the model never saw: ${count(c.test_size.sms).toLocaleString()} texts and ` +
    `${count(c.test_size.email).toLocaleString()} emails.`;
  $("tiles").innerHTML = [
    ["SMS macro-F1", pct(m.sms.macro_f1)],
    ["Email macro-F1", pct(m.email.macro_f1)],
    ["Real texts flagged (UCI/Mendeley)", pct(m.sms_uci_mendeley.legit_false_positive_rate, 2)],
    ["Real emails flagged", pct(m.email.legit_false_positive_rate, 2)],
    ["2023-25 phishing caught", pct(m.future_scam_recall)],
  ]
    .map(([k, v]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`)
    .join("");
  $("cmp").innerHTML =
    "<tr><th>Model</th><th>SMS macro-F1</th><th>Email macro-F1</th><th>2023-25 phishing</th><th>Size</th></tr>" +
    Object.entries(c.models)
      .sort((a, b) => b[1].sms.macro_f1 + b[1].email.macro_f1 - a[1].sms.macro_f1 - a[1].email.macro_f1)
      .map(
        ([n, s]) => `<tr class="${n === c.headline_model ? "best" : ""}"><td>${MODEL_NAMES[n] || n}</td>
        <td>${pct(s.sms.macro_f1)}</td><td>${pct(s.email.macro_f1)}</td><td>${pct(s.future_scam_recall)}</td><td>${s.size_mb ?? "n/a"} MB</td></tr>`,
      )
      .join("");
}

for (const b of document.querySelectorAll(".seg-btn")) b.addEventListener("click", () => setDomain(b.dataset.domain));
$("go").addEventListener("click", check);
$("msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) check();
});
$("examples").addEventListener("click", (e) => {
  const i = e.target.dataset.i;
  if (i === undefined) return;
  $("msg").value = EXAMPLES[domain][i][1];
  check();
});

setDomain("sms");
load().catch((err) => {
  $("load-text").textContent = "Couldn't load the model: " + err.message;
  $("load-bar").style.background = "var(--spam)";
});
