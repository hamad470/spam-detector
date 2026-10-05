const $ = (id) => document.getElementById(id);
const pct = (x) => (x * 100).toFixed(1) + "%";
// Naive Bayes is overconfident near 0 and 1; avoid claiming certainty.
const probPct = (x) => (x > 0.999 ? ">99.9%" : x < 0.001 ? "<0.1%" : pct(x));
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const LABELS = {
  bernoulli_nb: "Bernoulli NB",
  multinomial_nb: "Multinomial NB",
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
};

async function analyse() {
  const text = $("msg").value.trim();
  if (!text) return $("msg").focus();
  $("go").disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error((await res.json()).detail?.[0]?.msg || res.statusText);
    render(text, await res.json());
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    $("go").disabled = false;
  }
}

function render(text, r) {
  const spam = r.label === "spam";
  $("verdict").textContent = spam ? "Spam" : "Not spam";
  $("verdict").className = "verdict " + (spam ? "spam" : "ham");
  $("prob").textContent = `spam probability ${probPct(r.spam_probability)}`;
  $("bar").style.width = pct(r.spam_probability);
  $("bar").style.background = `var(--${spam ? "spam" : "ham"})`;
  $("meter-label").setAttribute("aria-label", `Spam probability ${pct(r.spam_probability)}`);

  const weights = Object.fromEntries(r.evidence.map((e) => [e.word, e.weight]));
  $("highlighted").innerHTML = esc(text).replace(/[A-Za-z0-9]+/g, (w) => {
    const wt = weights[w.toLowerCase()];
    return wt === undefined ? w : `<mark class="${wt > 0 ? "s" : "h"}" title="weight ${wt}">${w}</mark>`;
  });

  const max = Math.max(...r.evidence.map((e) => Math.abs(e.weight)), 1);
  $("evidence").innerHTML = r.evidence.length
    ? r.evidence
        .map((e) => `<li><span class="w">${esc(e.word)}</span>
          <span class="b" style="width:${(Math.abs(e.weight) / max) * 100}%;background:var(--${e.weight > 0 ? "spam" : "ham"})"></span>
          <span class="n">${e.weight > 0 ? "+" : ""}${e.weight.toFixed(2)}</span></li>`)
        .join("")
    : '<li class="muted">None of the words are in the model\'s 1,000-word vocabulary, so it falls back on the base rate.</li>';
  $("result").hidden = false;
}

async function loadCard() {
  const c = await (await fetch("/api/model")).json();
  const m = c.test_metrics;
  const d = c.dataset;
  $("model-desc").textContent =
    `${LABELS[c.model] || c.model} on TF-IDF features (top 1,000 by chi-squared). Trained on ${d.train_size.toLocaleString()} messages ` +
    `and evaluated on ${d.test_size.toLocaleString()} unseen ones. ${pct(d.spam_share)} of the dataset is spam.`;
  $("tiles").innerHTML = [["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1", m.f1], ["ROC-AUC", m.roc_auc]]
    .map(([k, v]) => `<div class="tile"><div class="v">${pct(v)}</div><div class="k">${k}</div></div>`)
    .join("");
  const cm = m.confusion_matrix;
  $("cm").innerHTML = `<tr><th></th><th>pred ham</th><th>pred spam</th></tr>
    <tr><td>actual ham</td><td class="hit">${cm.tn}</td><td class="miss">${cm.fp}</td></tr>
    <tr><td>actual spam</td><td class="miss">${cm.fn}</td><td class="hit">${cm.tp}</td></tr>`;
  $("cmp").innerHTML =
    "<tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th></tr>" +
    Object.entries(c.cv_comparison)
      .sort((a, b) => b[1].f1 - a[1].f1)
      .map(([n, s]) => `<tr class="${n === c.model ? "best" : ""}"><td>${LABELS[n] || n}</td><td>${pct(s.precision)}</td><td>${pct(s.recall)}</td><td>${pct(s.f1)}</td></tr>`)
      .join("");
}

$("go").addEventListener("click", analyse);
$("msg").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyse(); });
$("examples").addEventListener("click", (e) => {
  if (!e.target.dataset.text) return;
  $("msg").value = e.target.dataset.text;
  analyse();
});
loadCard().catch(() => ($("model-desc").textContent = "Model card unavailable."));

// Shareable links: /?text=... pre-fills and analyses the message.
const shared = new URLSearchParams(location.search).get("text");
if (shared) {
  $("msg").value = shared;
  analyse();
}
