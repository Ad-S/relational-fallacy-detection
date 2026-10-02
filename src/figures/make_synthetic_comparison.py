"""
Compare real vs synthetic examples on the two features that actually show a
meaningful distributional difference (explicit_denial_B rate, length ratio),
as a plausible (not causal) explanation for why adding synthetic data
slightly hurt the XGBoost model.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("figures")
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY, "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


with open("data/annotations/dataset_final.csv") as f:
    base = {r["pair_id"]: r for r in csv.DictReader(f)}
with open("data/annotations/dataset_final_specificity_features.csv") as f:
    spec = {r["pair_id"]: r for r in csv.DictReader(f)}

groups = {"reddit_cmv": [], "synthetic_handwritten": []}
for pid, b in base.items():
    if pid in spec:
        groups[b["source"]].append(spec[pid])

denial_rate = {src: sum(int(r["explicit_denial_B"]) for r in rows) / len(rows) for src, rows in groups.items()}
length_ratio = {src: [float(r["length_ratio_B_over_A"]) for r in rows] for src, rows in groups.items()}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes[0]
srcs = ["reddit_cmv", "synthetic_handwritten"]
labels = ["Real (CMV)", "Synthetic"]
colors = [BLUE, ORANGE]
rates = [denial_rate[s] for s in srcs]
bars = ax.bar(labels, rates, color=colors, width=0.5)
for rect, v in zip(bars, rates):
    ax.annotate(f"{v:.0%}", (rect.get_x() + rect.get_width()/2, v), xytext=(0, 4),
                textcoords="offset points", ha="center")
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.0)
ax.set_title("Rate of explicit denial phrase\nin claim_B (\"I never said...\")", fontsize=12)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
style_axis(ax)

ax = axes[1]
bp = ax.boxplot([length_ratio["reddit_cmv"], length_ratio["synthetic_handwritten"]],
                tick_labels=labels, patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
for median in bp["medians"]:
    median.set_color("white")
ax.set_ylabel("length(claim_B) / length(claim_A)")
ax.set_title("Claim length ratio, B over A", fontsize=12)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
style_axis(ax)

fig.suptitle("Real vs. synthetic: where the two sources differ", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "real_vs_synthetic_features.png", dpi=300, bbox_inches="tight")
print(f"Denial rate -- real: {denial_rate['reddit_cmv']:.3f}, synthetic: {denial_rate['synthetic_handwritten']:.3f}")
print("Wrote figures/real_vs_synthetic_features.png")
