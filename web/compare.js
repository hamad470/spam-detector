// "Fine-tuned model vs Gemini": draws random validation messages from compare.json (written by
// `python -m spam_detector.llm_compare`) and scores all three systems on the same draw.

const $ = (id) => document.getElementById(id);
const LABELS = ["legit", "spam", "phishing"];
const SYSTEMS = [
  ["model", "Fine-tuned DistilRoBERTa"],
  ["gemini_zero_shot", "Gemini, zero-shot prompt"],
  ["gemini_few_shot", "Gemini, few-shot prompt"],
];
const DRAW = 30;
const pct = (x, d = 1) => (x == null || Number.isNaN(x) ? "n/a" : (x * 100).toFixed(d) + "%");
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

let data;
let scope = "all";

// Same definitions as llm_compare.metrics(): an error or refusal counts as wrong
function metrics(rows, key) {
  const y = rows.map((r) => LABELS.indexOf(r.label));
  const p = rows.map((r) => LABELS.indexOf(r[key]));
  const pe = p.map((v, i) => (v < 0 ? (y[i] + 1) % 3 : v));
  const present = [...new Set(y)];
  const f1s = present.map((c) => {
    const tp = pe.filter((v, i) => v === c && y[i] === c).length;
    const fp = pe.filter((v, i) => v === c && y[i] !== c).length;
    const fn = pe.filter((v, i) => v !== c && y[i] === c).length;
    return tp === 0 ? 0 : (2 * tp) / (2 * tp + fp + fn);
  });
  const scam = y.map((v) => v > 0);
  const flagged = pe.map((v) => v > 0);
  const share = (mask, test) => {
    const idx = mask.map((m, i) => (m ? i : -1)).filter((i) => i >= 0);
    return idx.length ? idx.filter(test).length / idx.length : null;
  };
  return {
    accuracy: p.filter((v, i) => v === y[i]).length / y.length,
    macro: f1s.reduce((a, b) => a + b, 0) / f1s.length,
    caught: share(scam, (i) => flagged[i]),
    flagged: share(scam.map((s) => !s), (i) => flagged[i]),
  };
}

function draw() {
  let pool = data.messages.filter((m) => scope === "all" || m.domain === scope);
  if ($("cmp-disagree").checked) pool = pool.filter((m) => !(m.model === m.gemini_zero_shot && m.model === m.gemini_few_shot));
  const picked = [...pool].sort(() => Math.random() - 0.5).slice(0, DRAW);
  render(picked, pool.length);
}

function render(rows, poolSize) {
  const all = scope === "all" ? "all" : scope;
  const head = `<tr><th>System</th><th>Accuracy</th><th>Macro-F1</th><th>Scams caught</th><th>Real messages flagged</th>
    <th>Macro-F1, all ${data.summary.model[all].n} (95% CI)</th></tr>`;
  const scores = SYSTEMS.map(([key]) => metrics(rows, key));
  const best = Math.max(...scores.map((s) => s.accuracy));
  $("cmp-metrics").innerHTML =
    head +
    SYSTEMS.map(([key, name], i) => {
      const s = scores[i];
      const full = data.summary[key][all];
      return `<tr class="${s.accuracy === best ? "best" : ""}"><td>${name}</td><td>${pct(s.accuracy)}</td><td>${pct(s.macro)}</td>
        <td>${pct(s.caught)}</td><td>${pct(s.flagged)}</td>
        <td>${pct(full.macro_f1)} <span class="muted">(${pct(full.macro_f1_ci95[0], 0)}-${pct(full.macro_f1_ci95[1], 0)})</span></td></tr>`;
    }).join("");
  $("cmp-note").textContent =
    `${rows.length} messages drawn at random from ${poolSize}. With 30 messages one mistake moves accuracy by over 3 points, ` +
    `so draw a few times; the last column is the stable figure.`;

  const chip = (pred, truth) =>
    `<span class="chip-lbl ${pred}${pred === truth ? "" : " wrong"}">${pred === truth ? "✓" : "✗"} ${pred}</span>`;
  $("cmp-list").innerHTML = rows
    .map(
      (r) => `<li>
      <details>
        <summary><span class="dom">${r.domain === "sms" ? "SMS" : "Email"}</span>
          <span class="snippet">${esc(r.text.slice(0, 110))}${r.text.length > 110 ? "…" : ""}</span></summary>
        <p class="full">${esc(r.text)}</p>
      </details>
      <div class="preds"><span class="truth">true: <b>${r.label}</b></span>
        <span class="who">model</span>${chip(r.model, r.label)}
        <span class="who">zero-shot</span>${chip(r.gemini_zero_shot, r.label)}
        <span class="who">few-shot</span>${chip(r.gemini_few_shot, r.label)}</div>
    </li>`,
    )
    .join("");
}

function renderProbes() {
  const p = data.probes || [];
  if (!p.length) return;
  const mark = (pred, truth) => `<span class="chip-lbl ${pred}${pred === truth ? "" : " wrong"}">${pred === truth ? "✓" : "✗"} ${pred}</span>`;
  const score = (k) => p.filter((r) => r[k] === r.label).length;
  $("cmp-probes").innerHTML =
    `<tr><th>Message</th><th>True</th><th>Model (${score("model")}/${p.length})</th>` +
    `<th>Gemini zero-shot (${score("gemini_zero_shot")}/${p.length})</th><th>Gemini few-shot (${score("gemini_few_shot")}/${p.length})</th></tr>` +
    p
      .map(
        (r) => `<tr><td class="probe-text"><span class="dom">${r.domain === "sms" ? "SMS" : "Email"}</span> ${esc(r.text.split("\n")[0].slice(0, 90))}</td>
        <td>${r.label}</td><td>${mark(r.model, r.label)}</td><td>${mark(r.gemini_zero_shot, r.label)}</td><td>${mark(r.gemini_few_shot, r.label)}</td></tr>`,
      )
      .join("");
}

function setScope(d) {
  scope = d;
  for (const b of document.querySelectorAll(".cmp-dom")) {
    b.classList.toggle("active", b.dataset.dom === d);
    b.setAttribute("aria-checked", String(b.dataset.dom === d));
  }
  draw();
}

async function init() {
  const res = await fetch("compare.json");
  if (!res.ok) return; // comparison not generated yet: keep the section hidden
  data = await res.json();
  $("cmp-intro").textContent =
    `The same ${data.pool_size} held-out validation messages (half SMS, half email; 60 legit, 45 spam and 45 phishing per channel) ` +
    `were classified by the model on this page and by ${data.gemini_model}, prompted two ways and forced to answer with ` +
    `exactly one of legit, spam or phishing (structured output, temperature 0).`;
  $("cmp-prompts").innerHTML = Object.entries(data.prompts)
    .map(([v, ps]) => `<h4>${v.replace("_", "-")} (SMS)</h4><pre>${esc(ps.sms)}</pre>`)
    .join("");
  for (const b of document.querySelectorAll(".cmp-dom")) b.addEventListener("click", () => setScope(b.dataset.dom));
  $("cmp-draw").addEventListener("click", draw);
  $("cmp-disagree").addEventListener("change", draw);
  $("compare").hidden = false;
  renderProbes();
  draw();
}

init().catch(() => {});
