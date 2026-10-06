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

async function check() {
  const text = $("msg").value.trim();
  if (!text) return $("msg").focus();
  $("go").disabled = true;
  try {
    const res = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error((await res.json()).detail?.[0]?.msg || res.statusText);
    render(text, await res.json());
  } catch (err) {
    alert("Request failed: " + err.message);
  } finally {
    $("go").disabled = false;
  }
}

function render(text, r) {
  const c = CLASSES[r.label];
  $("verdict").textContent = c.title;
  $("verdict").className = "verdict " + c.cls;
  $("headline").textContent = `scam probability ${pct(r.scam_probability)}`;

  $("stack").innerHTML = Object.entries(r.probabilities)
    .map(([k, v]) => `<span class="seg ${k}" style="flex-grow:${Math.max(v, 0.002)}" title="${k} ${pct(v)}"></span>`)
    .join("");
  $("stack").setAttribute("aria-label", Object.entries(r.probabilities).map(([k, v]) => `${k} ${pct(v)}`).join(", "));
  $("legend").innerHTML = Object.entries(r.probabilities)
    .map(([k, v]) => `<span><i class="sw ${k}"></i>${k} <b>${pct(v)}</b></span>`)
    .join("");

  // Highlight by character span so repeated words are marked where they actually occur
  const spans = [...r.evidence].sort((a, b) => a.start - b.start);
  let html = "", pos = 0;
  for (const e of spans) {
    html += esc(text.slice(pos, e.start));
    html += `<mark class="${e.impact > 0 ? "s" : "h"}" title="impact ${e.impact}">${esc(text.slice(e.start, e.end))}</mark>`;
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
  $("result").hidden = false;
}

async function loadCard() {
  const c = await (await fetch("/api/model")).json();
  if (!c.final_model) throw new Error("no card");
  const m = c.models[c.final_model];
  $("model-desc").textContent =
    `${MODEL_NAMES[c.final_model]}, quantised to int8 ONNX (${c.onnx.onnx_size_mb} MB). ` +
    `Tested on ${c.test_messages.toLocaleString()} messages from spam campaigns it never saw during training.`;
  const o = c.onnx; // scores of the quantised model this page is actually running
  $("tiles").innerHTML = [
    ["Macro-F1 (3 classes)", pct(o.test_macro_f1)],
    ["Scams caught", pct(o.test_scam_recall)],
    ["Smishing recall", pct(o.test_smishing_recall)],
    ["Real texts flagged", pct(o.test_ham_false_positive_rate, 2)],
    ["Robust @30% obfuscation", pct(m.robustness.mixed["0.3"])],
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
$("msg").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) check(); });
$("examples").addEventListener("click", (e) => {
  if (!e.target.dataset.text) return;
  $("msg").value = e.target.dataset.text;
  check();
});
loadCard().catch(() => ($("model-desc").textContent = "Model card unavailable."));

const shared = new URLSearchParams(location.search).get("text");
if (shared) {
  $("msg").value = shared;
  check();
}
