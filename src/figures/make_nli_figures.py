"""
Visualize the NLI entailment scores driving the ablation result: a scatter
of entail(A->B) vs entail(B->A) per example, colored by label, plus the
asymmetry distribution (entail_AB - entail_BA) by class.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY, "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

LABEL_COLORS = {"NONE": TEXT_SECONDARY, "MOTTE_BAILEY": BLUE, "STRAWMAN": ORANGE}
LABEL_ORDER = ["NONE", "MOTTE_BAILEY", "STRAWMAN"]


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def load_data():
    with open("data/annotations/dataset_final.csv") as f:
        base = {r["pair_id"]: r for r in csv.DictReader(f)}
    with open("data/annotations/dataset_final_nli_features.csv") as f:
        nli = {r["pair_id"]: r for r in csv.DictReader(f)}

    rows = []
    for pid, b in base.items():
        if pid in nli:
            n = nli[pid]
            rows.append({
                "pair_id": pid, "label": b["label"], "type": b["type_candidate"],
                "entail_AB": float(n["entail_AB"]), "entail_BA": float(n["entail_BA"]),
            })
    return rows


def fig_scatter(rows):
    fig, ax = plt.subplots(figsize=(6, 6))
    for label in LABEL_ORDER:
        xs = [r["entail_AB"] for r in rows if r["label"] == label]
        ys = [r["entail_BA"] for r in rows if r["label"] == label]
        ax.scatter(xs, ys, label=label, color=LABEL_COLORS[label], alpha=0.75, s=45,
                   edgecolors="white", linewidths=0.5)

    ax.plot([0, 1], [0, 1], linestyle="--", color=GRID, linewidth=1.2, zorder=0)
    ax.set_xlabel("Entailment(A → B)")
    ax.set_ylabel("Entailment(B → A)")
    ax.set_title("Bidirectional entailment by label")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nli_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_asymmetry_hist(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    bins = [i / 20 for i in range(-20, 21)]

    for ax, label in zip(axes, LABEL_ORDER):
        asym = [r["entail_AB"] - r["entail_BA"] for r in rows if r["label"] == label]
        ax.hist(asym, bins=bins, color=LABEL_COLORS[label], edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.set_title(f"{label} (n={len(asym)})", fontsize=12)
        ax.set_xlabel("Entail(A→B) − Entail(B→A)")
        style_axis(ax)

    axes[0].set_ylabel("Count")
    fig.suptitle("NLI asymmetry by label", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nli_asymmetry_by_label.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_data()
    fig_scatter(rows)
    fig_asymmetry_hist(rows)
    print(f"Wrote NLI figures to {OUT_DIR}/ (n={len(rows)} examples)")
