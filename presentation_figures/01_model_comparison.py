"""Fig 1: XGBoost vs zero-shot LLM on the 91 real examples (accuracy and F1).

Bars   = point estimates, recomputed from the prediction CSVs; they match the
         mid-report's Table 3 (asserted against reported_mid_report_tables.csv).
Whiskers + paired test = AUDIT-DERIVED, not in the report: percentile bootstrap
         (10,000 resamples of the 91 examples, same resample for both models, seed 0).

Run from anywhere:  venv/bin/python presentation_figures/01_model_comparison.py
Needs: data/annotations/dataset_consolidated_xgboost_oof_predictions.csv
       (regenerate with: venv/bin/python src/models/ablation.py dataset_consolidated)
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from common import *

B, SEED = 10_000, 0
MODELS = [("xgb", "XGBoost (NLI + specificity features)", C_XGB),
          ("llm", "Zero-shot LLM (Claude Sonnet 5)", C_LLM)]

style()
df = load_real_predictions()
rep = reported_tables()
rng = np.random.default_rng(SEED)

rows, paired = [], []
for shape, pos, name in SHAPES:
    sub = df[df.type_candidate == shape].reset_index(drop=True)
    y = sub.gold.values
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))            # shared resamples -> paired comparison
    maj = majority_rate(y)
    boots = {}
    for key, _, _ in MODELS:
        p = sub[key].values
        m = metrics(y, p)
        acc_b = (p[idx] == y[idx]).mean(axis=1)
        f1_b = np.array([metrics(y[i], p[i])["f1"] for i in idx])
        boots[key] = acc_b
        r = dict(shape=shape, model=key, n=n, majority_rate=maj, accuracy=m["accuracy"], f1=m["f1"],
                 precision=m["precision"], recall=m["recall"],
                 accuracy_ci_low=np.percentile(acc_b, 2.5), accuracy_ci_high=np.percentile(acc_b, 97.5),
                 f1_ci_low=np.percentile(f1_b, 2.5), f1_ci_high=np.percentile(f1_b, 97.5))
        rep_row = rep[(rep["shape"] == shape) & (rep["model"] == ("XGBoost" if key == "xgb" else "Zero-shot LLM"))].iloc[0]
        for col in ("accuracy", "precision", "recall", "f1"):
            assert round(r[col], 3) == round(rep_row[col], 3), (shape, key, col, r[col], rep_row[col])
        r["matches_report_table3"] = True
        rows.append(r)
    diff = boots["llm"] - boots["xgb"]
    b = int(((sub.llm == y) & (sub.xgb != y)).sum())    # LLM right, XGBoost wrong
    c = int(((sub.llm != y) & (sub.xgb == y)).sum())
    paired.append(dict(shape=shape, n=n, acc_diff_llm_minus_xgb=(sub.llm == y).mean() - (sub.xgb == y).mean(),
                       diff_ci_low=np.percentile(diff, 2.5), diff_ci_high=np.percentile(diff, 97.5),
                       llm_only_correct=b, xgb_only_correct=c, mcnemar_exact_p=mcnemar_exact(b, c)))

stats = pd.DataFrame(rows)
pairs = pd.DataFrame(paired)
stats.to_csv(DER / "model_comparison_stats.csv", index=False)
pairs.to_csv(DER / "model_comparison_paired_tests.csv", index=False)

# ------------------------------------------------------------------ figure --
fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharey=True)
fig.subplots_adjust(left=0.07, right=0.985, top=0.72, bottom=0.25, wspace=0.08)
W = 0.34
for ax, (metric, title) in zip(axes, [("accuracy", "Accuracy"), ("f1", "F1 (positive class)")]):
    for gi, (shape, pos, name) in enumerate(SHAPES):
        for mi, (key, _, color) in enumerate(MODELS):
            r = stats[(stats["shape"] == shape) & (stats.model == key)].iloc[0]
            x = gi + (mi - 0.5) * (W + 0.03)
            ax.bar(x, r[metric], W, color=color, zorder=3)
            ax.errorbar(x, r[metric], yerr=[[r[metric] - r[f"{metric}_ci_low"]], [r[f"{metric}_ci_high"] - r[metric]]],
                        color=INK, capsize=5, lw=1.6, capthick=1.6, zorder=4)
            ax.text(x, 0.03, f"{r[metric]:.2f}", ha="center", va="bottom", color="white",
                    fontsize=15, fontweight="bold", zorder=5)
        if metric == "accuracy":
            maj = stats[stats["shape"] == shape].majority_rate.iloc[0]
            ax.plot([gi - 0.46, gi + 0.46], [maj, maj], ls=(0, (4, 3)), color=INK2, lw=1.6, zorder=2)
    ax.set_xticks(range(len(SHAPES)))
    ax.set_xticklabels([f"{name}\nn={int(stats[stats['shape'] == s].n.iloc[0])}, majority {stats[stats['shape'] == s].majority_rate.iloc[0]:.2f}"
                        for s, _, name in SHAPES])
    ax.set_xlim(-0.6, len(SHAPES) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, loc="left", pad=12)
    clean_axis(ax)
axes[0].set_ylabel("Score")
axes[1].tick_params(labelleft=False)

handles = [Patch(color=c, label=l) for _, l, c in MODELS]
handles += [Line2D([0], [0], color=INK, lw=1.6, marker="_", markersize=10, label="95% bootstrap CI  (new, not in report)"),
            Line2D([0], [0], color=INK2, lw=1.6, ls=(0, (4, 3)), label="majority-class accuracy")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.935), ncol=2, frameon=False, columnspacing=1.8)
fig.suptitle("Zero-shot LLM vs XGBoost on the 91 real examples", x=0.07, y=0.985, ha="left",
             fontsize=19, fontweight="bold")

p = {r["shape"]: r for _, r in pairs.iterrows()}
mb, sm = p["motte_bailey"], p["strawman"]
footer(fig, [
    "Bars = mid-report Table 3 values, recomputed from the prediction CSVs (exact match). Whiskers and the gaps below are audit-derived: 10,000 paired bootstrap resamples, seed 0.",
    f"Accuracy gap LLM - XGBoost: motte-and-bailey {mb.acc_diff_llm_minus_xgb:+.2f} [{mb.diff_ci_low:+.2f}, {mb.diff_ci_high:+.2f}], McNemar exact p={mb.mcnemar_exact_p:.2f};  "
    f"strawman {sm.acc_diff_llm_minus_xgb:+.2f} [{sm.diff_ci_low:+.2f}, {sm.diff_ci_high:+.2f}], p={sm.mcnemar_exact_p:.3f}.",
    "XGBoost: real-only run, 5-fold stratified CV, seed 42. LLM: single run, same model that extracted the claims, prompt not archived -> not an independent baseline.",
])
save(fig, "01_model_comparison.png")
