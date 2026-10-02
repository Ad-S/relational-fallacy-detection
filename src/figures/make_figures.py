"""
Generate publication-ready static figures (PNG, 300 DPI) for the mid-submission
report. Uses the validated categorical/sequential palette from the dataviz
reference (references/palette.md): blue=#2a78d6, orange=#eb6834.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axis(ax, hide_spines=("top", "right")):
    for s in hide_spines:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------
# 1. Model comparison: F1 per shape, XGBoost vs Zero-shot (both on 91 real)
# ---------------------------------------------------------------------------
def fig_model_comparison():
    shapes = ["Motte-bailey", "Strawman"]
    xgb_f1 = [0.579, 0.591]
    zs_f1 = [0.750, 0.800]

    x = range(len(shapes))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - width/2 for i in x], xgb_f1, width, label="XGBoost (NLI + specificity features)", color=BLUE)
    b2 = ax.bar([i + width/2 for i in x], zs_f1, width, label="Zero-shot LLM", color=ORANGE)

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, color=TEXT_PRIMARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels(shapes)
    ax.set_ylabel("F1 (positive class)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fallacy-vs-NONE F1 by model and shape\n(n=91 real examples)", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Dataset construction funnel (log scale, direct-labeled)
# ---------------------------------------------------------------------------
def fig_dataset_funnel():
    stages = [
        ("Raw CMV utterances", 293_297),
        ("Reply-shape candidates", 376_714),
        ("Keyword-filtered candidates", 5_078),
        ("Sampled for annotation", 300),
        ("Usable after claim extraction", 240),
        ("Hand-labeled (final, incl. synthetic)", 111),
    ]
    labels = [s[0] for s in stages][::-1]
    values = [s[1] for s in stages][::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=BLUE, height=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_title("From raw corpus to labeled dataset")

    for rect, v in zip(bars, values):
        ax.annotate(f"{v:,}", (rect.get_width(), rect.get_y() + rect.get_height()/2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=TEXT_PRIMARY)

    ax.grid(axis="x", which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_funnel.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Label distribution, real vs synthetic (stacked bar)
# ---------------------------------------------------------------------------
def fig_label_distribution():
    labels = ["NONE", "MOTTE_BAILEY", "STRAWMAN"]

    with open("data/annotations/dataset_final.csv") as f:
        rows = list(csv.DictReader(f))
    real_counts = Counter(r["label"] for r in rows if r["source"] == "reddit_cmv")
    synth_counts = Counter(r["label"] for r in rows if r["source"] == "synthetic_handwritten")
    real = [real_counts[l] for l in labels]
    synthetic = [synth_counts[l] for l in labels]
    print(f"Label distribution -- real: {dict(zip(labels, real))}, synthetic: {dict(zip(labels, synthetic))}")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    b1 = ax.bar(labels, real, label="Real (CMV)", color=BLUE)
    b2 = ax.bar(labels, synthetic, bottom=real, label="Synthetic (hand-written)", color=ORANGE)

    for i, lbl in enumerate(labels):
        total = real[i] + synthetic[i]
        ax.annotate(f"{total}", (i, total), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, color=TEXT_PRIMARY)

    ax.set_ylabel("Count")
    ax.set_title("Final label distribution (n=111)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "label_distribution.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_model_comparison()
    fig_dataset_funnel()
    fig_label_distribution()
    print(f"Wrote figures to {OUT_DIR}/")
