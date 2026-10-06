"""Draw every figure in the README and report from reports/results.json.

python -m spam_detector.figures
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from .paths import ARTIFACTS, FIGURES, LABELS, REPORTS
from .paths import MODELS as MODELS_DIR

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
# Fixed slot per model so a colour always means the same model across figures
MODELS = {
    "distilroberta_aug": ("DistilRoBERTa + attack aug.", "#2a78d6"),
    "distilroberta": ("DistilRoBERTa", "#eb6834"),
    "minilm_lr": ("MiniLM embeddings + LR", "#1baf7a"),
    "tfidf_lr": ("TF-IDF word+char + LR", "#eda100"),
    "nb_original": ("Original Naive Bayes", "#e87ba4"),
}
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
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


def data_audit(audit):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})

    ax = axes[0]
    copied, new = audit["mendeley_rows_copied_from_uci"], audit["mendeley_rows_genuinely_new"]
    ax.barh(["Mendeley 2022"], [copied], color="#86b6ef", height=0.5)
    ax.barh(["Mendeley 2022"], [new], left=[copied + 25], color="#2a78d6", height=0.5)
    ax.text(copied / 2, 0, f"{copied:,} already in UCI 2011", ha="center", va="center", color=INK)
    ax.text(copied + new / 2 + 25, 0, f"{new}\nnew", ha="center", va="center", color="white")
    ax.set_title(f"{copied / (copied + new):.0%} of the 2022 dataset is the 2011 dataset")
    ax.set_xlabel("unique messages")
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    nd = audit["near_duplicate_clusters"]
    shares = [nd["share_of_ham_in_clusters"], nd["share_of_spam_or_smishing_in_clusters"]]
    bars = ax.bar(["ham", "spam + smishing"], shares, color=["#86b6ef", "#2a78d6"], width=0.5)
    for b, s in zip(bars, shares, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.015, f"{s:.0%}", ha="center", color=INK)
    ax.set_ylim(0, 0.62)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Messages that belong to a\nnear-duplicate campaign")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    final = pd.DataFrame(audit["final"]).T[LABELS].sum()
    bars = ax.bar(LABELS, final.values, color=["#86b6ef", "#2a78d6", "#104281"], width=0.5)
    for b, v in zip(bars, final.values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{int(v):,}", ha="center", color=INK)
    ax.set_title("Final labelled corpus")
    ax.set_ylim(0, final.max() * 1.15)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "data_audit.png")


def leakage(results):
    lk = results["leakage"]
    rows = [
        ("Naive merge,\nrandom split", lk["naive_merge_random_split"]),
        ("Exact de-dup,\nrandom split", lk["exact_dedup_random_split"]),
        ("Campaign-grouped split\n(this project)", lk["near_dup_grouped_split"]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(rows))[::-1]
    vals = [r[1]["f1_mean"] for r in rows]
    colors = ["#cde2fb", "#86b6ef", "#2a78d6"]
    errors = [r[1]["f1_std"] for r in rows]
    ax.barh(y, vals, xerr=errors, color=colors, height=0.55, error_kw={"ecolor": INK_2, "capsize": 3})
    for yi, v, e in zip(y, vals, errors, strict=True):
        ax.text(v + e + 0.002, yi, f"{v:.3f}", va="center", color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(min(vals) - 0.05, 1.0)
    ax.set_xlabel("spam-vs-ham F1, same TF-IDF + LR model")
    ax.set_title("The same model looks better when test messages leak into training")
    ax.grid(axis="y", visible=False)
    leaked = lk["naive_test_messages_with_exact_copy_in_train"]
    note = f"Under the naive protocol {leaked:.0%} of test messages have an exact copy in the training set."
    ax.text(0, -0.28, note, transform=ax.transAxes, color=INK_2, fontsize=9)
    _save(fig, "leakage.png")


def model_comparison(results):
    names = _present(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, key, title in [
        (axes[0], "macro", "3-class macro-F1 (ham / spam / smishing)"),
        (axes[1], "binary", "Scam vs ham F1 (spam + smishing)"),
    ]:
        for yi, n in zip(y, names, strict=True):
            r = results["models"][n]
            v, ci = (
                (r["macro_f1"], r["macro_f1_ci95"])
                if key == "macro"
                else (r["binary"]["f1"], r["binary"]["f1_ci95"])
            )
            ax.barh(yi, v, color=MODELS[n][1], height=0.6)
            ax.errorbar(v, yi, xerr=[[v - ci[0]], [ci[1] - v]], color=INK, capsize=3, lw=1)
            ax.text(ci[1] + 0.006, yi, f"{v:.3f}", va="center", color=INK)
        ax.set_title(title)
        ax.set_xlim(0.6 if key == "macro" else 0.75, 1.0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [MODELS[n][0] for n in names])
    fig.text(
        0.01,
        -0.03,
        "Error bars: 95% bootstrap confidence intervals over the campaign-held-out test set.",
        color=INK_2,
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def confusion(results):
    pair = [n for n in ["nb_original", "distilroberta_aug"] if n in results["models"]]
    fig, axes = plt.subplots(1, len(pair), figsize=(4.6 * len(pair), 4))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, pair, strict=True):
        cm = np.array(results["models"][n]["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else INK, fontsize=10)  # fmt: skip
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.grid(False)
        ax.set_title(MODELS[n][0])
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def pr_curves(results, preds):
    y = (preds.label != "ham").to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for n in _present(results):
        p, r, _ = precision_recall_curve(y, preds[f"{n}__malicious"])
        ax.plot(
            r,
            p,
            color=MODELS[n][1],
            label=f"{MODELS[n][0]}  (AP {results['models'][n]['binary']['pr_auc']:.3f})",
        )
    ax.set_xlabel("recall (share of scams caught)")
    ax.set_ylabel("precision")
    ax.set_xlim(0.5, 1.005)
    ax.set_ylim(0.5, 1.01)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Precision-recall, scam vs ham")
    _save(fig, "pr_curves.png")


def robustness(results):
    attacks = ["leetspeak", "homoglyph", "spacing", "typo", "mixed"]
    fig, axes = plt.subplots(1, len(attacks), figsize=(16, 3.6), sharey=True)
    ablation = results.get("ablation_tfidf_lr_without_normaliser", {})
    for ax, attack in zip(axes, attacks, strict=True):
        for n in _present(results):
            curve = results["models"][n]["robustness"][attack]
            rates = [float(k) for k in curve]
            ax.plot(rates, list(curve.values()), color=MODELS[n][1], marker="o", ms=4, label=MODELS[n][0])
        if attack in ablation:
            curve = ablation[attack]
            ax.plot([float(k) for k in curve], list(curve.values()), color=MODELS["tfidf_lr"][1], ls="--",
                    lw=1.5, label="TF-IDF + LR, no normaliser")  # fmt: skip
        ax.set_title(attack)
        ax.set_xlabel("share of words obfuscated")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("scams still caught (recall)")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle("Robustness to obfuscation attacks", x=0.01, ha="left", fontweight="semibold", color=INK)
    fig.tight_layout()
    _save(fig, "robustness.png")


def calibration(results, preds):
    y = (preds.label != "ham").to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":", label="perfectly calibrated")
    for n in [m for m in ["nb_original", "tfidf_lr", "distilroberta_aug"] if m in results["models"]]:
        frac, mean = calibration_curve(y, preds[f"{n}__malicious"], n_bins=8, strategy="quantile")
        ece = results["models"][n]["binary"]["ece"]
        ax.plot(mean, frac, color=MODELS[n][1], marker="o", ms=5, label=f"{MODELS[n][0]}  (ECE {ece:.3f})")
    ax.set_xlabel("predicted scam probability")
    ax.set_ylabel("observed share of scams")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Reliability diagram")
    _save(fig, "calibration.png")


def tradeoff(results):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n in _present(results):
        r = results["models"][n]
        size = max(r["size_mb"], 0.5)
        ax.scatter(
            r["latency_ms"],
            r["macro_f1"],
            s=40 + size * 3,
            color=MODELS[n][1],
            edgecolor=SURFACE,
            lw=2,
            zorder=3,
        )
        label = f"{MODELS[n][0]}\n{r['size_mb']} MB"
        point = (r["latency_ms"], r["macro_f1"])
        ax.annotate(label, point, textcoords="offset points", xytext=(10, -4), fontsize=9, color=INK)
    card_path = MODELS_DIR / "onnx" / "metrics.json"
    if card_path.exists():  # the model that actually ships: same network, int8 ONNX
        onnx = json.loads(card_path.read_text())["onnx"]
        src = results["models"]["distilroberta_aug"]
        point = (onnx["latency_ms"], onnx["test_macro_f1"])
        ax.scatter(
            *point, s=40 + onnx["onnx_size_mb"] * 3, facecolor=SURFACE, edgecolor="#2a78d6", lw=2.5, zorder=3
        )
        ax.annotate("", point, (src["latency_ms"], src["macro_f1"]),
                    arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 1})  # fmt: skip
        label = f"deployed: int8 ONNX\n{onnx['onnx_size_mb']} MB"
        ax.annotate(
            label, point, textcoords="offset points", xytext=(16, -34), fontsize=9, color=INK, ha="left"
        )
    ax.set_xscale("log")
    ax.set_xlabel("latency per message, ms (log scale, laptop CPU)")
    ax.set_ylabel("3-class macro-F1")
    ax.set_title("Accuracy vs cost")
    ax.margins(x=0.35, y=0.15)
    _save(fig, "tradeoff.png")


def training_curves():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for n in ["distilroberta_aug", "distilroberta"]:
        path = ARTIFACTS / n / "history.json"
        if path.exists():
            h = json.loads(path.read_text())
            ax.plot(
                [e["epoch"] for e in h],
                [e["val_macro_f1"] for e in h],
                color=MODELS[n][1],
                marker="o",
                label=MODELS[n][0],
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Fine-tuning")
    ax.legend(fontsize=9)
    _save(fig, "training_curves.png")


def main():
    results = json.loads((REPORTS / "results.json").read_text())
    audit = json.loads((REPORTS / "data_audit.json").read_text())
    preds = pd.read_parquet(REPORTS / "test_predictions.parquet")
    data_audit(audit)
    leakage(results)
    model_comparison(results)
    confusion(results)
    pr_curves(results, preds)
    robustness(results)
    calibration(results, preds)
    tradeoff(results)
    training_curves()


if __name__ == "__main__":
    main()
