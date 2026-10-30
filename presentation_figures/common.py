"""Shared helpers for the presentation figures.

All plotted values are computed from the CSVs in data/annotations/. The only
transcribed numbers live in reported_mid_report_tables.csv (mid-report Tables 3
and 4); they are used to cross-check the recomputed values and to label which
numbers were already in the report -- never as plot input.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANNOT = ROOT / "data" / "annotations"
OUT = ROOT / "presentation_figures"
FIG = OUT / "figures"
DER = OUT / "derived"
FIG.mkdir(exist_ok=True)
DER.mkdir(exist_ok=True)

SHAPES = [("motte_bailey", "MOTTE_BAILEY", "Motte-and-bailey"),
          ("strawman", "STRAWMAN", "Strawman")]
SHAPE_NAME = {s: n for s, _, n in SHAPES}
POS_LABEL = {s: p for s, p, _ in SHAPES}

# Palette: reference categorical slots 1-3 (validated all-pairs, light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL_BAR = "#a9a8a1"   # "NONE" / non-model category
C_XGB, C_LLM = BLUE, ORANGE


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 13,
        "legend.fontsize": 13, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    })


def clean_axis(ax, grid_axis="y"):
    ax.tick_params(length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def footer(fig, lines, y0=0.012, size=10.5):
    """Small provenance text at the bottom of the figure (bottom-up)."""
    for i, line in enumerate(reversed(lines)):
        fig.text(0.01, y0 + i * 0.034, line, fontsize=size, color=INK2, ha="left", va="bottom")


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# ----------------------------------------------------------------- metrics --
def confusion(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(tn=int(((y == 0) & (p == 0)).sum()), fp=int(((y == 0) & (p == 1)).sum()),
                fn=int(((y == 1) & (p == 0)).sum()), tp=int(((y == 1) & (p == 1)).sum()))


def metrics(y, p):
    c = confusion(y, p)
    n = len(y)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(n=n, accuracy=(c["tp"] + c["tn"]) / n, precision=prec, recall=rec, f1=f1, **c)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# -------------------------------------------------------------------- data --
def load_real_predictions():
    """One row per real example with gold, real-only XGBoost OOF, and LLM prediction (all binary).

    Sources (no values hard-coded):
      gold      <- dataset_consolidated.csv (label) cross-checked against both prediction files
      XGBoost   <- dataset_consolidated_xgboost_oof_predictions.csv (real-only run, 5-fold CV, seed 42)
      zero-shot <- zeroshot_baseline_predictions.csv
    """
    gold = pd.read_csv(ANNOT / "dataset_consolidated.csv")[["pair_id", "type_candidate", "label"]]
    xgb = pd.read_csv(ANNOT / "dataset_consolidated_xgboost_oof_predictions.csv")
    llm = pd.read_csv(ANNOT / "zeroshot_baseline_predictions.csv")
    assert len(gold) == len(xgb) == len(llm) == 91, (len(gold), len(xgb), len(llm))
    assert set(gold.pair_id) == set(xgb.pair_id) == set(llm.pair_id), "pair_id sets differ"
    df = gold.merge(xgb[["pair_id", "true_y", "pred_y"]], on="pair_id") \
             .merge(llm[["pair_id", "predicted_label", "true_label"]], on="pair_id")
    df["positive_label"] = df.type_candidate.map(POS_LABEL)
    df["gold"] = (df.label == df.positive_label).astype(int)
    assert (df.gold == df.true_y).all(), "gold disagrees with XGBoost OOF file"
    assert (df.label == df.true_label).all(), "gold disagrees with zero-shot file"
    bad = ~df.predicted_label.isin(["NONE"]) & (df.predicted_label != df.positive_label)
    assert not bad.any(), "LLM predicted a label belonging to the other shape"
    df["xgb"] = df.pred_y.astype(int)
    df["llm"] = (df.predicted_label == df.positive_label).astype(int)
    return df[["pair_id", "type_candidate", "label", "gold", "xgb", "llm"]]


def reported_tables():
    return pd.read_csv(OUT / "reported_mid_report_tables.csv")


def majority_rate(y):
    y = np.asarray(y)
    return max(y.mean(), 1 - y.mean())
