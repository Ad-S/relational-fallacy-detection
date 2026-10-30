"""Fig 3 (EXPLORATORY): rate of the explicit_denial_B indicator by source, candidate shape and label.

Counts come straight from dataset_final.csv joined with dataset_final_specificity_features.csv.
Wilson 95% intervals are audit-derived (not in the report). Descriptive only: this is NOT a test
of whether the models use the feature, and the masking ablation has not been run.

explicit_denial_B is a regex (src/features/specificity_features.py) matching: "I never said",
"I didn't/did not say", "I'm/am not saying", "I wasn't saying", "that's not what I said".
It does not include strawman trigger phrases ("so you're saying", "are you saying"), so it is 0
for every strawman-shaped row by construction.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import *

style()
d = pd.read_csv(ANNOT / "dataset_final.csv").merge(
    pd.read_csv(ANNOT / "dataset_final_specificity_features.csv")[["pair_id", "explicit_denial_B"]], on="pair_id")
assert len(d) == 111 and d.explicit_denial_B.isin([0, 1]).all()

g = (d.groupby(["source", "type_candidate", "label"]).explicit_denial_B.agg(k="sum", n="count").reset_index())
g["rate"] = g.k / g.n
g[["ci_low", "ci_high"]] = [wilson(int(k), int(n)) for k, n in zip(g.k, g.n)]
g.to_csv(DER / "explicit_denial_rates.csv", index=False)

SOURCES = [("reddit_cmv", "Real (CMV)"), ("synthetic_handwritten", "Synthetic (hand-written)")]
ROWS = [("motte_bailey", "MOTTE_BAILEY"), ("motte_bailey", "NONE"), ("strawman", "STRAWMAN"), ("strawman", "NONE")]

fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
fig.subplots_adjust(left=0.15, right=0.985, top=0.74, bottom=0.22, wspace=0.08)
for ax, (src, title) in zip(axes, SOURCES):
    ypos = [3.2, 2.2, 0.9, -0.1]
    for y, (shape, lab) in zip(ypos, ROWS):
        r = g[(g.source == src) & (g.type_candidate == shape) & (g.label == lab)]
        if r.empty:
            ax.text(0.02, y, "no examples", va="center", fontsize=12, color=MUTED, style="italic")
            continue
        r = r.iloc[0]
        color = NEUTRAL_BAR if lab == "NONE" else AQUA
        ax.barh(y, r.rate, 0.62, color=color, zorder=3)
        ax.errorbar(r.rate, y, xerr=[[r.rate - r.ci_low], [r.ci_high - r.rate]], color=INK, capsize=4, lw=1.5, zorder=4)
        ax.text(max(r.ci_high, r.rate) + 0.03, y, f"{int(r.k)}/{int(r.n)}  ({r.rate:.0%})", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of examples where B matches the denial regex")
    ax.set_title(title, loc="left", pad=10)
    clean_axis(ax, "x")
axes[0].set_yticks([3.2, 2.2, 0.9, -0.1])
axes[0].set_yticklabels(["MOTTE_BAILEY", "NONE", "STRAWMAN", "NONE"])
for ax in axes:
    ax.set_ylim(-0.75, 4.25)
for ax in axes:
    ax.text(0.0, 3.80, "Motte-and-bailey-shaped candidates", fontsize=13, fontweight="bold", color=INK2, va="bottom")
    ax.text(0.0, 1.36, "Strawman-shaped candidates", fontsize=13, fontweight="bold", color=INK2, va="bottom")
fig.legend(handles=[Patch(color=AQUA, label="fallacy label"), Patch(color=NEUTRAL_BAR, label="NONE")],
           loc="upper left", bbox_to_anchor=(0.15, 0.915), ncol=2, frameon=False)
fig.suptitle("Exploratory: how often does B contain an explicit denial phrase?", x=0.02, y=0.985, ha="left",
             fontsize=19, fontweight="bold")
real = d[d.source == "reddit_cmv"]
footer(fig, [
    "Descriptive counts from dataset_final.csv + dataset_final_specificity_features.csv; whiskers = Wilson 95% intervals (audit-derived). Not evidence of a shortcut by itself.",
    "The regex covers 'I never said', 'I'm not saying', 'that's not what I said' etc. but not strawman triggers ('so you're saying'), so strawman-shaped rows are 0 by construction.",
    f"Real data overall: {int(real.explicit_denial_B.sum())}/{len(real)} ({real.explicit_denial_B.mean():.1%}); synthetic: {int(d[d.source != 'reddit_cmv'].explicit_denial_B.sum())}/{(d.source != 'reddit_cmv').sum()}. Masking ablation not yet run.",
])
save(fig, "03_explicit_denial_rates.png")
