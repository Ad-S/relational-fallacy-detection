"""Fig 8: two synthetic examples - annotated label vs zero-shot LLM label (one agreement, one disagreement).

Text and gold labels: data/annotations/dataset_final.csv.
Zero-shot labels and reasons: presentation_figures/llm_rerun/zeroshot_rerun_predictions.csv
(the only zero-shot predictions that exist for the synthetic examples).
"""
import sys
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from common import *

# usage: 08_synthetic_examples.py [id1,id2] [output.png]   (default: one agreement, one disagreement)
IDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["syn003", "syn011"]
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "08_synthetic_examples.png"
GOOD, BAD = "#0ca30c", "#d03b3b"   # status colours, always paired with a text label

style()
gold = pd.read_csv(ANNOT / "dataset_final.csv").set_index("pair_id")
pred = pd.read_csv(OUT / "llm_rerun" / "zeroshot_rerun_predictions.csv").set_index("pair_id")
assert all(gold.loc[i, "source"] == "synthetic_handwritten" for i in IDS)

W_IN, H_IN = 13.33, 7.5
fig = plt.figure(figsize=(W_IN, H_IN))
fig.text(0.03, 0.955, "Synthetic examples: annotated label vs zero-shot LLM", fontsize=21, fontweight="bold", va="top")

LINE = 0.0335          # line height in figure fraction at 12.5pt
COL_W, GAP, X0 = 0.455, 0.03, 0.03
for ci, pid in enumerate(IDS):
    g, p = gold.loc[pid], pred.loc[pid]
    x = X0 + ci * (COL_W + GAP)
    top, y = 0.865, 0.865
    shape_name = SHAPE_NAME[g.type_candidate]
    fig.text(x + 0.015, y - 0.012, f"{pid}  ·  {shape_name}-shaped", fontsize=14, fontweight="bold", color=INK2, va="top")
    y -= 0.065
    fields = [("A", "Claim A", g.claim_A), ("C", "Challenge", g.challenge), ("B", "Claim B", g.claim_B)]
    for tag, name, text in fields:
        if not isinstance(text, str) or not text.strip():
            continue
        fig.text(x + 0.015, y, name, fontsize=12, fontweight="bold", color=INK2, va="top")
        lines = textwrap.wrap(text.strip(), 47)
        fig.text(x + 0.11, y, "\n".join(lines), fontsize=12.5, color=INK, va="top", linespacing=1.35)
        y -= LINE * len(lines) + 0.022
    y -= 0.01
    fig.add_artist(plt.Line2D([x + 0.015, x + COL_W - 0.015], [y, y], color=GRID, lw=1.2, transform=fig.transFigure))
    y -= 0.03
    agree = g.label == p.predicted_label
    for lab, val in [("Annotated label", g.label), ("Zero-shot LLM", p.predicted_label)]:
        fig.text(x + 0.015, y, lab, fontsize=13, color=INK2, va="center")
        fig.text(x + 0.17, y, val, fontsize=15, fontweight="bold", color=INK, va="center", family="monospace")
        y -= 0.05
    fig.add_artist(plt.Line2D([x + 0.022], [y], marker="o", markersize=13, color=GOOD if agree else BAD, transform=fig.transFigure))
    fig.text(x + 0.037, y, "Labels agree" if agree else "Labels disagree", fontsize=13, fontweight="bold", color=INK, va="center")
    y -= 0.05
    lines = textwrap.wrap(f"LLM's reason: “{p.reason}”", 76)
    fig.text(x + 0.015, y, "\n".join(lines), fontsize=11.5, color=INK2, va="top", style="italic", linespacing=1.3)
    y -= 0.028 * len(lines) + 0.02
    card = FancyBboxPatch((x, y), COL_W, top - y, boxstyle="round,pad=0,rounding_size=0.012",
                          transform=fig.transFigure, facecolor="white", edgecolor=GRID, lw=1.4, zorder=-1)
    fig.add_artist(card)

save(fig, OUT_NAME)
