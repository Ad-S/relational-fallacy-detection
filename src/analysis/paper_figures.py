"""Paper figures: one message per figure, minimal text, the same colour for a model in every figure.

Reads the files written by final_results.py / llm_results.py plus the raw predictions.
Usage: python3 src/analysis/paper_figures.py   (run from src/analysis or anywhere)
Writes: results_final/figures/fig*.pdf and .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import final_results as F

OUT = F.OUT / "figures"
OUT.mkdir(exist_ok=True)
LLM = F.ROOT / "llm_bundle" / "llm_results" / "llm"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
C = {"ft": "#2a78d6", "plain": "#86b6ef", "xgb": "#1baf7a", "q32": "#eb6834", "q7": "#f4b08f",
     "bonly": "#a9a8a1", "real": "#2a78d6", "hand": "#1baf7a", "var": "#a9a8a1"}
SH = {"motte_bailey": "Motte-and-bailey", "strawman": "Strawman"}
POS = {"motte_bailey": "MOTTE_BAILEY", "strawman": "STRAWMAN"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "savefig.bbox": "tight",
                     "figure.dpi": 150})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", name)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def zeroshot(folder):
    z = pd.read_csv(LLM / folder / "zeroshot.csv", keep_default_na=False)
    return z


def fig_roc(p):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), sharey=True)
    for ax, (shape, name) in zip(axes, SH.items()):
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        for cfg, col, lab in [("xgb_NLI + Specificity", C["xgb"], "XGBoost"), ("plain_pair", C["plain"], "DeBERTa"),
                              ("nli_pair", C["ft"], "DeBERTa-NLI")]:
            g = p[(p.config == cfg) & (p["shape"] == shape)]
            fpr, tpr, _ = roc_curve(g.y, g.prob)
            ax.plot(fpr, tpr, color=col, lw=1.6, label=lab)
        for folder, col, lab, mk in [("Qwen2.5-7B-Instruct", C["q7"], "Qwen-7B", "s"), ("Qwen2.5-32B-Instruct", C["q32"], "Qwen-32B", "o")]:
            z = zeroshot(folder)
            g = z[z.type_candidate == shape]
            y, pr = (g.label == POS[shape]).values, (g.predicted_label == POS[shape]).values
            tpr, fpr = (pr & y).sum() / y.sum(), (pr & ~y).sum() / (~y).sum()
            ax.scatter([fpr], [tpr], s=34, color=col, marker=mk, zorder=5, edgecolor="white", lw=0.8, clip_on=False,
                       label=lab + " (zero-shot)")
        ax.set_title(name)
        ax.set_xlabel("False positive rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        grid(ax, "both")
    axes[0].set_ylabel("True positive rate")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig1_roc")


def fig_reliance(t1, mask):
    """(a) input ablation: B only -> A,C,B (AUC); (b) trigger masking on trigger-bearing rows (accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2))
    for ax, title, rows, ylab in [
        (axes[0], "Input: B only vs. full exchange", [("nli_bonly", "B only"), ("nli_pair", "A, C, B")], "ROC-AUC"),
        (axes[1], "Trigger phrase removed from B", [("orig", "original"), ("masked", "masked")], "Accuracy"),
    ]:
        for shape, mk in [("motte_bailey", "o"), ("strawman", "s")]:
            if ax is axes[0]:
                vals = [t1[(t1.config == c) & (t1["shape"] == shape)].auc_mean.iloc[0] for c, _ in rows]
            else:
                r = mask[(mask.config == "nli_pair") & (mask["shape"] == shape)].iloc[0]
                vals = [r.acc_unmasked_changed, r.acc_masked_changed]
            ax.plot([0, 1], vals, color=C["ft"], lw=1.4, marker=mk, ms=5, alpha=1 if shape == "motte_bailey" else 0.55)
            ax.annotate(SH[shape], (1, vals[1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        ax.set_xticks([0, 1]); ax.set_xticklabels([l for _, l in rows])
        ax.set_xlim(-0.25, 1.75)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        grid(ax)
    axes[0].set_ylim(0.55, 0.8); axes[1].set_ylim(0.55, 0.7)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig2_what_the_model_uses")


def fig_probing(pr, t1):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.1), sharey=True)
    for ax, (shape, name) in zip(axes, SH.items()):
        for model, col, lab in [("deberta_plain", C["plain"], "DeBERTa (frozen)"), ("deberta_nli", C["ft"], "DeBERTa-NLI (frozen)")]:
            s = pr[(pr.model == model) & (pr["shape"] == shape) & (pr.metric == "auc")].sort_values("layer")
            ax.plot(s.layer, s["mean"], color=col, lw=1.5, label=lab)
        ft = t1[(t1.config == "nli_pair") & (t1["shape"] == shape)].auc_mean.iloc[0]
        ax.axhline(ft, color=C["ft"], ls="--", lw=1, label="DeBERTa-NLI (fine-tuned)")
        ax.axhline(0.5, color=INK2, ls=":", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Layer")
        grid(ax)
    axes[0].set_ylabel("Probe ROC-AUC")
    axes[0].set_ylim(0.45, 0.8)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig3_probing")


def fig_learning(lc, t1):
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0), sharey=True)
    for ax, (shape, name) in zip(axes, SH.items()):
        s = lc[lc["shape"] == shape].sort_values("frac")
        ax.errorbar(s.frac * 100, s.auc_mean, yerr=s.auc_std, color=C["ft"], marker="o", ms=4, lw=1.5, capsize=2,
                    label="DeBERTa-NLI")
        xg = t1[(t1.config == "xgb_NLI + Specificity") & (t1["shape"] == shape)].auc_mean.iloc[0]
        ax.axhline(xg, color=C["xgb"], ls="--", lw=1, label="XGBoost (all data)")
        ax.set_xticks([25, 50, 75, 100]); ax.set_xticklabels(["25%", "50%", "75%", "100%"])
        ax.set_title(name)
        ax.set_xlabel("Training data")
        grid(ax)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_ylim(0.45, 0.82)
    axes[1].legend(loc="lower right")
    save(fig, "fig4_learning_curve")


def fig_by_source(p):
    data = F.data
    srcmap = {"reddit_cmv": "Real", "handwritten": "Hand-written", "handwritten_variant": "Generated", "real_variant": "Generated"}
    cats = ["Real", "Hand-written", "Generated"]
    rows = []
    for cfg, lab, col in [("xgb_NLI + Specificity", "XGBoost", C["xgb"]), ("nli_pair", "DeBERTa-NLI", C["ft"])]:
        g = p[p.config == cfg].assign(S=lambda d: d.src.map(srcmap))
        acc = g.groupby(["S", "seed"]).apply(lambda h: (h.y == h.pred).mean(), include_groups=False).groupby("S").mean()
        rows.append((lab, col, [acc[c] for c in cats]))
    z = zeroshot("Qwen2.5-32B-Instruct").merge(data[["pair_id", "source"]], on="pair_id", suffixes=("", "_d"))
    z["S"] = z.source_d.map(srcmap)
    z["ok"] = z.predicted_label.where(z.predicted_label != "", "NONE") == z.label
    rows.append(("Qwen-32B (zero-shot)", C["q32"], [z[z.S == c].ok.mean() for c in cats]))
    n = data.source.map(srcmap).value_counts()
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.25
    for i, (lab, col, vals) in enumerate(rows):
        ax.bar(np.arange(3) + (i - 1) * (w + 0.02), vals, w, color=col, label=lab)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{c}\n(n={n[c]})" for c in cats])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 0.8)
    grid(ax)
    ax.legend(loc="upper left", ncol=1)
    save(fig, "fig5_by_source")


def fig_two_stage():
    t = pd.read_csv(F.OUT / "table9_two_stage.csv")
    keep = t[(t["shape"] == "strawman") | (t.rule == "denied X entailed (false denial)")]
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    w = 0.34
    for i, (ext, col) in enumerate([("Qwen2.5-7B", C["q7"]), ("Qwen2.5-32B", C["q32"])]):
        vals = [keep[(keep.extractor == ext) & (keep["shape"] == s)].auc.iloc[0] for s in SH]
        ax.bar(np.arange(2) + (i - 0.5) * (w + 0.03), vals, w, color=col, label=f"{ext.replace('Qwen2.5-', 'Qwen-')} extracts")
    ax.axhline(0.5, color=INK2, ls=":", lw=0.9)
    ax.set_xticks(range(2)); ax.set_xticklabels(list(SH.values()))
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.3, 0.75)
    grid(ax)
    ax.legend(loc="upper left")
    save(fig, "fig6_two_stage")


def fig_dataset():
    d = F.data
    srcmap = {"reddit_cmv": "Real (CMV)", "handwritten": "Hand-written", "handwritten_variant": "Generated", "real_variant": "Generated"}
    d = d.assign(S=d.source.map(srcmap))
    order = ["Real (CMV)", "Hand-written", "Generated"]
    labs = [("MOTTE_BAILEY", "#6250d6", "Motte-and-bailey"), ("STRAWMAN", "#e87ba4", "Strawman"), ("NONE", "#d9d8d2", "None")]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    for yi, s in enumerate(order):
        left = 0
        for lab, col, name in labs:
            k = int(((d.S == s) & (d.label == lab)).sum())
            ax.barh(yi, k, 0.6, left=left, color=col, edgecolor="white", lw=1, label=name if yi == 0 else None)
            if k >= 20:
                ax.text(left + k / 2, yi, str(k), ha="center", va="center", fontsize=7, color="white" if lab == "MOTTE_BAILEY" else INK)
            left += k
        ax.text(left + 6, yi, str(int((d.S == s).sum())), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(3)); ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Examples")
    grid(ax, "x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=3, handlelength=1, columnspacing=1)
    save(fig, "fig0_dataset")


def fig_variants_help():
    t = pd.read_csv(F.OUT / "table11_originals_only.csv")
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    w = 0.34
    for i, (train, col, lab) in enumerate([("originals only", C["plain"], "Trained on originals only"),
                                           ("originals + 475 variants", C["ft"], "+ generated variants")]):
        g = t[(t.training == train) & (t.model == "DeBERTa-NLI")]
        vals = [g[g["shape"] == s].auc_mean.iloc[0] for s in SH]
        errs = [g[g["shape"] == s].auc_std.iloc[0] for s in SH]
        ax.bar(np.arange(2) + (i - 0.5) * (w + 0.03), vals, w, yerr=errs, color=col, label=lab,
               error_kw=dict(lw=0.8, capsize=2))
    ax.axhline(0.5, color=INK2, ls=":", lw=0.9)
    ax.set_xticks(range(2)); ax.set_xticklabels(list(SH.values()))
    ax.set_ylabel("ROC-AUC on original examples")
    ax.set_ylim(0.3, 0.8)
    grid(ax)
    ax.legend(loc="upper left")
    save(fig, "fig7_variants_help")


def main():
    p = F.load()
    t1 = pd.read_csv(F.OUT / "table1_main.csv")
    mask = pd.read_csv(F.OUT / "table3_masking.csv")
    lc = pd.read_csv(F.OUT / "table6_learning_curve.csv")
    pr = pd.read_csv(F.RES / "probing" / "probe_results.csv")
    fig_dataset()
    fig_roc(p)
    fig_reliance(t1, mask)
    fig_probing(pr, t1)
    fig_learning(lc, t1)
    fig_by_source(p)
    fig_two_stage()
    fig_variants_help()


if __name__ == "__main__":
    main()
