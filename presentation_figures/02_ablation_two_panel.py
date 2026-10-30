"""Fig 2: feature ablation, real-only (left) vs real + synthetic (right).

Right panel = mid-report Table 6 (dataset_final_ablation_results.csv, 111 examples).
Left panel  = NEW, regenerated for this presentation with the existing, unmodified script
              src/models/ablation.py (venv/bin/python src/models/ablation.py dataset_consolidated)
              -> data/annotations/dataset_consolidated_ablation_results.csv (91 real examples).
Same XGBoost settings and 5-fold stratified CV (seed 42) in both. No intervals: the ablation
scripts keep only the full-feature OOF predictions, so per-feature-set uncertainty is not
available without changing the experiment code.

Produces 02_ablation_accuracy.png and 02_ablation_f1.png.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from common import *

SETS = [("NLI only", AQUA), ("Specificity only", NEUTRAL_BAR), ("NLI + Specificity", BLUE)]
SET_LABEL = {"NLI only": "NLI features only", "Specificity only": "Specificity features only",
             "NLI + Specificity": "NLI + specificity (full model)"}
PANELS = [("dataset_consolidated", "Real only", "NEW - regenerated for this talk"),
          ("dataset_final", "Real + synthetic", "mid-report Table 6")]

style()
tables, major = {}, {}
for stem, _, _ in PANELS:
    t = pd.read_csv(ANNOT / f"{stem}_ablation_results.csv")
    labels = pd.read_csv(ANNOT / f"{stem}.csv")
    tables[stem] = t
    for shape, pos, _ in SHAPES:
        y = (labels[labels.type_candidate == shape].label == pos).astype(int)
        major[(stem, shape)] = (majority_rate(y), len(y))
        assert (t[t["shape"] == shape].n == len(y)).all(), (stem, shape)

long = []
for stem, _, _ in PANELS:
    for _, r in tables[stem].iterrows():
        long.append(dict(dataset=stem, **r.to_dict(), majority_rate=major[(stem, r["shape"])][0]))
pd.DataFrame(long).to_csv(DER / "ablation_long.csv", index=False)


def draw(metric, fname, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.70, bottom=0.25, wspace=0.07)
    W = 0.26
    for ax, (stem, title, tag) in zip(axes, PANELS):
        t = tables[stem]
        for gi, (shape, pos, name) in enumerate(SHAPES):
            for si, (fs, color) in enumerate(SETS):
                v = t[(t["shape"] == shape) & (t.feature_set == fs)][metric].iloc[0]
                x = gi + (si - 1) * (W + 0.03)
                ax.bar(x, v, W, color=color, zorder=3)
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK, zorder=5)
            if metric == "accuracy":
                m = major[(stem, shape)][0]
                ax.plot([gi - 0.47, gi + 0.47], [m, m], ls=(0, (4, 3)), color=INK2, lw=1.6, zorder=4)
        ax.set_xticks(range(len(SHAPES)))
        ax.set_xticklabels([f"{name}\nn={major[(stem, s)][1]}, majority {major[(stem, s)][0]:.2f}" for s, _, name in SHAPES])
        ax.set_xlim(-0.6, len(SHAPES) - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{title}", loc="left", pad=26)
        ax.text(0, 1.035, tag, transform=ax.transAxes, fontsize=12, color=INK2, va="bottom")
        clean_axis(ax)
    axes[0].set_ylabel(ylabel)
    axes[1].tick_params(labelleft=False)
    handles = [Patch(color=c, label=SET_LABEL[n]) for n, c in SETS]
    if metric == "accuracy":
        handles.append(Line2D([0], [0], color=INK2, lw=1.6, ls=(0, (4, 3)), label="majority-class accuracy"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.925), ncol=2, frameon=False, columnspacing=1.8)
    fig.suptitle(f"Feature ablation: {ylabel}", x=0.065, y=0.985, ha="left", fontsize=19, fontweight="bold")
    n_small = min(v[1] for v in major.values())
    footer(fig, [
        "XGBoost (50 trees, depth 3), 5-fold stratified CV, seed 42, same in both panels. One CV run; no intervals shown.",
        f"With n=44-58 per task, one example is worth {100/max(v[1] for v in major.values()):.1f}-{100/n_small:.1f} accuracy points; differences of a few points are within noise.",
        "Motte-and-bailey, NLI-only vs full model (" + metric + "): " + "; ".join(
            f"{lab} {tables[stem][(tables[stem]['shape'] == 'motte_bailey') & (tables[stem].feature_set == 'NLI only')][metric].iloc[0]:.2f} vs "
            f"{tables[stem][(tables[stem]['shape'] == 'motte_bailey') & (tables[stem].feature_set == 'NLI + Specificity')][metric].iloc[0]:.2f}"
            for stem, lab, _ in PANELS) + ".",
    ])
    save(fig, fname)


draw("accuracy", "02_ablation_accuracy.png", "Accuracy")
draw("f1", "02_ablation_f1.png", "F1 (positive class)")
