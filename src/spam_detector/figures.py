"""Draw every figure in the README and report from reports/*.json.

python -m spam_detector.figures
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from .paths import DOMAINS, FIGURES, LABELS, PROCESSED, REPORTS

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
# Fixed colour per model, so a colour means the same model in every figure
MODELS = {
    "distilroberta": ("DistilRoBERTa (fine-tuned)", "#2a78d6"),
    "tfidf_lr": ("TF-IDF word+char + LR", "#eda100"),
    "nb_original": ("Original Naive Bayes", "#e87ba4"),
}
CHANNEL = {"sms": "SMS", "email": "Email"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def _present(results):
    return [m for m in MODELS if m in results["models"]]


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _pct_axis(ax, axis="y"):
    fmt = lambda v, _: f"{v:.0%}"  # noqa: E731
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def data_overview(corpus: pd.DataFrame, audit: dict):
    """Where the messages come from, and how much of each source survives de-duplication."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    names = {
        "uci+mendeley": "UCI 2011 / Mendeley 2022",
        "mendeley_2022": "Mendeley 2022 (new)",
        "uci_only": "UCI 2011 (only)",
        "uci_2011": "UCI 2011 (only)",
        "exais_2014": "ExAIS (Nigeria)",
        "enron": "Enron-Spam",
        "spamassassin": "SpamAssassin",
        "nazario": "Nazario phishing",
    }
    colors = {"legit": "#86b6ef", "spam": "#eda100", "phishing": "#c8322b", "spam_unlabelled": "#d9d6cc"}
    ax = axes[0]
    counts = (
        corpus.assign(src=corpus.source.map(names)).groupby(["src", "label"]).size().unstack(fill_value=0)
    )
    counts = counts.loc[counts.sum(1).sort_values().index]
    left = np.zeros(len(counts))
    for label in ["legit", "spam", "phishing", "spam_unlabelled"]:
        if label in counts:
            ax.barh(
                counts.index,
                counts[label],
                left=left,
                color=colors[label],
                label=label.replace("_", " "),
                height=0.6,
            )
            left += counts[label].to_numpy()
    for y, total in enumerate(left):
        ax.text(total + 200, y, f"{int(total):,}", va="center", color=INK, fontsize=9)
    ax.set_title("Unique messages per source after de-duplication")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.15)

    ax = axes[1]
    c = audit["campaigns"]
    x = np.arange(2)
    w = 0.25
    for i, label in enumerate(["legit", "spam", "phishing"]):
        vals = [c[f"{d}_{label}_share_in_campaign"] for d in DOMAINS]
        bars = ax.bar(x + (i - 1) * w, vals, width=w - 0.03, color=colors[label], label=label)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontsize=8.5, color=INK
            )
    ax.set_xticks(x, ["SMS", "Email"])
    _pct_axis(ax)
    ax.set_ylim(0, 0.75)
    ax.set_title("Share of messages that belong to a\nnear-duplicate campaign")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "data_overview.png")


def shortcut_audit(audit: dict):
    stages = [
        ("as_packaged", "Datasets as downloaded"),
        ("raw", "+ Enron's tokenised\nformatting undone"),
        ("scrubbed", "+ addresses, names, list\nfooters and years removed"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(stages))[::-1]
    unseen = [audit[k]["flagged_share_spamassassin_legit_unseen_source"] for k, _ in stages]
    seen = [audit[k]["flagged_share_enron_legit_seen_source"] for k, _ in stages]
    ax.barh(
        y + 0.18,
        unseen,
        height=0.34,
        color="#c8322b",
        label="legit emails from an unseen source (SpamAssassin)",
    )
    ax.barh(
        y - 0.18, seen, height=0.34, color="#86b6ef", label="legit emails from the training source (Enron)"
    )
    for yi, u, s in zip(y, unseen, seen, strict=True):
        ax.text(u + 0.01, yi + 0.18, f"{u:.1%}", va="center", fontsize=9, color=INK)
        ax.text(s + 0.01, yi - 0.18, f"{s:.1%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [s for _, s in stages])
    _pct_axis(ax, "x")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("legitimate emails wrongly flagged as phishing")
    ax.set_title("Same model, same data: removing corpus fingerprints")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("What a linear model leans on most for 'phishing'")
    rows = [
        ("As downloaded", audit["as_packaged"]["top_phishing_features"][:6]),
        ("After clean-up", audit["scrubbed"]["top_phishing_features"][:6]),
    ]
    for i, (name, feats) in enumerate(rows):
        ax.text(0, 0.82 - i * 0.42, name, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(
            0,
            0.68 - i * 0.42,
            ",  ".join(feats),
            color=INK_2,
            transform=ax.transAxes,
            wrap=True,
            fontsize=9.5,
        )
    ax.text(0, -0.02, "jose@monkey.org is the inbox the phishing corpus was collected from.",
            color=INK_2, fontsize=8.5, transform=ax.transAxes)  # fmt: skip
    fig.tight_layout()
    _save(fig, "shortcut_audit.png")


def leakage(results):
    lk = results["leakage"]
    rows = [
        ("Naive merge,\nrandom split", lk["naive_merge_random_split"]),
        ("Exact de-dup,\nrandom split", lk["exact_dedup_random_split"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = np.arange(len(rows))[::-1]
    vals = [r[1]["f1_mean"] for r in rows]
    errs = [r[1]["f1_std"] for r in rows]
    ax.barh(
        y,
        vals,
        xerr=errs,
        color=["#cde2fb", "#2a78d6"],
        height=0.55,
        error_kw={"ecolor": INK_2, "capsize": 3},
    )
    for yi, v, e in zip(y, vals, errs, strict=True):
        ax.text(v + e + 0.002, yi, f"{v:.3f}", va="center", color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(min(vals) - 0.04, 1.0)
    ax.set_xlabel("SMS scam F1, TF-IDF + LR (3 random splits)")
    ax.set_title("UCI + Mendeley: exact copies inflate the score")
    ax.grid(axis="y", visible=False)
    copies = lk["naive_test_messages_with_exact_copy_in_train"]
    note = f"Naive protocol: {copies:.0%} of test messages also appear in training."
    ax.text(0, -0.42, note, transform=ax.transAxes, color=INK_2, fontsize=9)
    _save(fig, "leakage.png")


def model_comparison(results):
    names = _present(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, d in zip(axes, DOMAINS, strict=True):
        for yi, n in zip(y, names, strict=True):
            m = results["models"][n][d]
            v, (lo, hi) = m["macro_f1"], m["macro_f1_ci95"]
            ax.barh(yi, v, color=MODELS[n][1], height=0.6)
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color=INK, capsize=3, lw=1)
            ax.text(hi + 0.008, yi, f"{v:.3f}", va="center", color=INK)
        ax.set_title(f"{CHANNEL[d]}: 3-class macro-F1 (legit / spam / phishing)")
        ax.set_xlim(0.3 if d == "sms" else 0.7, 1.0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [MODELS[n][0] for n in names])
    fig.text(0.01, -0.05, "Error bars: 95% bootstrap intervals over the campaign-held-out test set. "
             "DistilRoBERTa: mean of 3 training seeds.", color=INK_2, fontsize=9)  # fmt: skip
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def confusion(results):
    m = results["models"].get("distilroberta")
    if not m:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, d in zip(axes, DOMAINS, strict=True):
        cm = np.array(m[d]["confusion_matrix"], dtype=float)  # mean over seeds
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.0f}\n{norm[i, j]:.0%}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else INK, fontsize=9.5)  # fmt: skip
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.grid(False)
        ax.set_title(f"{CHANNEL[d]}, DistilRoBERTa")
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def transfer(results):
    tr = results["transfer"]
    models = [
        m for m in ("tfidf_lr", "distilroberta") if tr.get(m) and all(tr[m].get(k) for k in ("sms", "email"))
    ]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.4))
    axes = np.atleast_1d(axes)
    rows = [("sms", "SMS only"), ("email", "Email only"), ("sms-email", "Both")]
    for ax, m in zip(axes, models, strict=True):
        grid = np.array([[tr[m][r][d] for d in DOMAINS] for r, _ in rows])
        ax.imshow(grid, cmap=BLUES, vmin=0.3, vmax=1)
        for i in range(3):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if grid[i, j] > 0.8 else INK,
                )
        ax.set_xticks(range(2), ["tested on\nSMS", "tested on\nemail"])
        ax.set_yticks(range(3), [f"trained on {r}" for _, r in rows])
        ax.grid(False)
        ax.set_title(MODELS[m][0])
    fig.suptitle("Macro-F1 across channels", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "transfer.png")


def phishing_over_time(results):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for n in _present(results):
        m = results["models"][n]
        eras = m["test_phishing_by_era"]
        future = m["future"]["by_year"]
        xs = ["2005-07", "2015-22"] + [str(y) for y in future]
        ys = [eras["2005-2007"], eras["2015-2022"]] + list(future.values())
        ax.plot(xs, ys, marker="o", color=MODELS[n][1], label=MODELS[n][0])
    ax.axvspan(1.5, len(xs) - 0.5, color="#f0efec", zorder=0)
    ax.text(1.6, 0.03, "after the training data ends", color=INK_2, fontsize=9)
    _pct_axis(ax)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("phishing emails flagged as scams")
    ax.set_title("Does it catch phishing written after its training data?")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "phishing_over_time.png")


def no_hook(results):
    names = _present(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, d in zip(axes, DOMAINS, strict=True):
        x = np.arange(len(names))
        w = 0.36
        with_h = [results["models"][n]["no_hook"][d]["scam_recall_with_hooks"] for n in names]
        without = [results["models"][n]["no_hook"][d]["scam_recall"] for n in names]
        n_without = int(results["models"][names[0]]["no_hook"][d]["scams"])
        ax.bar(
            x - w / 2, with_h, width=w - 0.03, color="#86b6ef", label="has a link, number, address or amount"
        )
        ax.bar(x + w / 2, without, width=w - 0.03, color="#c8322b", label="none")
        for xi, a, b in zip(x, with_h, without, strict=True):
            ax.text(xi - w / 2, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
            ax.text(xi + w / 2, b + 0.02, f"{b:.0%}", ha="center", fontsize=9)
        ax.set_xticks(x, [MODELS[n][0].replace(" + ", "\n+ ") for n in names], fontsize=8.5)
        _pct_axis(ax)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{CHANNEL[d]}: scams caught ({n_without} test scams have no hook)")
        ax.grid(axis="x", visible=False)
    handles, _ = axes[0].get_legend_handles_labels()
    names = ["has a link, number, address or amount", "has none of them"]
    fig.legend(handles, names, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.tight_layout()
    _save(fig, "no_hook.png")


def robustness(results):
    attacks = ["leetspeak", "homoglyph", "spacing", "typo", "mixed"]
    rob = results["robustness"]
    fig, axes = plt.subplots(2, len(attacks), figsize=(16, 6.4), sharey=True, sharex=True)
    for row, d in enumerate(DOMAINS):
        for ax, attack in zip(axes[row], attacks, strict=True):
            for n in _present(results):
                if n in rob:
                    curve = rob[n][d][attack]
                    xs, ys = [float(k) for k in curve], list(curve.values())
                    ax.plot(xs, ys, color=MODELS[n][1], marker="o", ms=4, label=MODELS[n][0])
            if "tfidf_lr_without_normaliser" in rob:
                curve = rob["tfidf_lr_without_normaliser"][d][attack]
                ax.plot([float(k) for k in curve], list(curve.values()), color=MODELS["tfidf_lr"][1], ls="--",
                        lw=1.5, label="TF-IDF + LR, no normaliser")  # fmt: skip
            ax.set_title(f"{CHANNEL[d]} · {attack}", fontsize=11)
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set_ylim(0, 1.03)
            if row == 1:
                ax.set_xlabel("share of words disguised")
        axes[row][0].set_ylabel("scams still caught")
        _pct_axis(axes[row][0])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle(
        "Robustness to obfuscation (validation scams)", x=0.01, ha="left", fontweight="bold", color=INK
    )
    fig.tight_layout()
    _save(fig, "robustness.png")


def calibration_and_pr(results, preds: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
    for col, d in enumerate(DOMAINS):
        part = preds[(preds.split == "test") & (preds.domain == d) & preds.label.isin(LABELS)]
        y = (part.label != "legit").to_numpy()
        ax_pr, ax_cal = axes[0][col], axes[1][col]
        ax_cal.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":", label="perfectly calibrated")
        for n in _present(results):
            col_name = f"{n}__p_legit"
            if col_name not in part:
                continue
            score = 1 - part[col_name].to_numpy()
            p, r, _ = precision_recall_curve(y, score)
            ax_pr.plot(
                r, p, color=MODELS[n][1], label=f"{MODELS[n][0]} (AP {results['models'][n][d]['pr_auc']:.3f})"
            )
            frac, mean = calibration_curve(y, score, n_bins=8, strategy="quantile")
            ax_cal.plot(mean, frac, color=MODELS[n][1], marker="o", ms=4,
                        label=f"{MODELS[n][0]} (ECE {results['models'][n][d]['ece']:.3f})")  # fmt: skip
        ax_pr.set_xlim(0.5, 1.005)
        ax_pr.set_ylim(0.5, 1.01)
        ax_pr.set_xlabel("recall (share of scams caught)")
        ax_pr.set_ylabel("precision")
        ax_pr.set_title(f"{CHANNEL[d]}: precision-recall, scam vs legit")
        ax_pr.legend(loc="lower left", fontsize=8.5)
        ax_cal.set_xlabel("predicted scam probability")
        ax_cal.set_ylabel("observed share of scams")
        ax_cal.set_title(f"{CHANNEL[d]}: reliability")
        ax_cal.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "pr_and_calibration.png")


def training_curves(results):
    hist = results.get("training_history")
    if not hist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, d in zip(axes, DOMAINS, strict=True):
        for i, (name, h) in enumerate(hist.items()):
            epochs, scores = [e["epoch"] for e in h], [e[f"val_f1_{d}"] for e in h]
            color = MODELS["distilroberta"][1]
            ax.plot(
                epochs,
                scores,
                marker="o",
                color=color,
                alpha=0.45 + 0.25 * i,
                label=name.replace("sms-email_", ""),
            )
        ax.set_xlabel("epoch")
        ax.set_title(f"{CHANNEL[d]}: validation macro-F1")
        ax.set_xticks([1, 2, 3])
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "training_curves.png")


def main():
    results = json.loads((REPORTS / "results.json").read_text())
    audit = json.loads((REPORTS / "data_audit.json").read_text())
    shortcuts = json.loads((REPORTS / "shortcut_audit.json").read_text())
    corpus = pd.read_parquet(PROCESSED / "corpus.parquet", columns=["source", "label"])
    preds = pd.read_parquet(REPORTS / "test_predictions.parquet")
    data_overview(corpus, audit)
    shortcut_audit(shortcuts)
    leakage(results)
    model_comparison(results)
    confusion(results)
    transfer(results)
    phishing_over_time(results)
    no_hook(results)
    robustness(results)
    calibration_and_pr(results, preds)
    training_curves(results)


if __name__ == "__main__":
    main()
