"""Fig 6: zero-shot LLM vs NLI-only XGBoost vs full XGBoost, on the 91 real examples.

Only the real set is possible: the zero-shot LLM was never run on the 20 synthetic examples
(excluded in the original run because their labels were visible in the same session; mid-report
sec 5), so no real + synthetic LLM comparison can be plotted without a new LLM run.

NLI-only out-of-fold predictions are not saved by src/models/ablation.py, so they are recomputed
here with that script's own run_one() (same features, XGBoost settings, folds, seed) and asserted
equal to data/annotations/dataset_consolidated_ablation_results.csv. Full-model predictions come
from dataset_consolidated_xgboost_oof_predictions.csv, LLM predictions from
zeroshot_baseline_predictions.csv. Whiskers: paired bootstrap 95% CI (10,000 resamples, seed 0),
audit-derived, not in the report.
"""
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import *

spec = importlib.util.spec_from_file_location("ablation", ROOT / "src" / "models" / "ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)

B, SEED = 10_000, 0
MODELS = [("nli", "XGBoost, NLI features only", AQUA),
          ("xgb", "XGBoost, NLI + specificity (full)", C_XGB),
          ("llm", "Zero-shot LLM (Claude Sonnet 5)", C_LLM)]

style()
df = load_real_predictions()
feats = pd.read_csv(ANNOT / "dataset_consolidated.csv") \
    .merge(pd.read_csv(ANNOT / "dataset_consolidated_nli_features.csv"), on="pair_id") \
    .merge(pd.read_csv(ANNOT / "dataset_consolidated_specificity_features.csv"), on="pair_id")
saved_abl = pd.read_csv(ANNOT / "dataset_consolidated_ablation_results.csv")

nli_preds = {}
for shape, pos, _ in SHAPES:
    sub = feats[feats.type_candidate == shape].copy()
    sub["y"] = (sub.label == pos).astype(int)
    m, oof = ablation.run_one(sub, ablation.NLI_FEATURES, sub["pair_id"], save_oof=True)
    ref = saved_abl[(saved_abl["shape"] == shape) & (saved_abl.feature_set == "NLI only")].iloc[0]
    for col in ("accuracy", "precision", "recall", "f1"):
        assert round(m[col], 3) == round(ref[col], 3), (shape, col, m[col], ref[col])
    nli_preds.update(dict(zip(oof.pair_id, oof.pred_y)))
df["nli"] = df.pair_id.map(nli_preds).astype(int)

rng = np.random.default_rng(SEED)
rows = []
for shape, pos, name in SHAPES:
    sub = df[df.type_candidate == shape].reset_index(drop=True)
    y = sub.gold.values
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    for key, _, _ in MODELS:
        p = sub[key].values
        m = metrics(y, p)
        acc_b = (p[idx] == y[idx]).mean(axis=1)
        f1_b = np.array([metrics(y[i], p[i])["f1"] for i in idx])
        rows.append(dict(shape=shape, model=key, n=n, majority_rate=majority_rate(y),
                         accuracy=m["accuracy"], f1=m["f1"], precision=m["precision"], recall=m["recall"],
                         accuracy_ci_low=np.percentile(acc_b, 2.5), accuracy_ci_high=np.percentile(acc_b, 97.5),
                         f1_ci_low=np.percentile(f1_b, 2.5), f1_ci_high=np.percentile(f1_b, 97.5)))
    for a, b in [("llm", "nli")]:
        bw = int(((sub[a] == y) & (sub[b] != y)).sum())
        cw = int(((sub[a] != y) & (sub[b] == y)).sum())
        rows[-1].update({f"mcnemar_p_{a}_vs_{b}": mcnemar_exact(bw, cw),
                         f"{a}_only_correct_vs_{b}": bw, f"{b}_only_correct_vs_{a}": cw})
stats = pd.DataFrame(rows)
stats.to_csv(DER / "llm_vs_nli_real_stats.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6), sharey=True)
fig.subplots_adjust(left=0.065, right=0.985, top=0.70, bottom=0.25, wspace=0.08)
W = 0.25
for ax, (metric, title) in zip(axes, [("accuracy", "Accuracy"), ("f1", "F1 (positive class)")]):
    for gi, (shape, pos, name) in enumerate(SHAPES):
        for mi, (key, _, color) in enumerate(MODELS):
            r = stats[(stats["shape"] == shape) & (stats.model == key)].iloc[0]
            x = gi + (mi - 1) * (W + 0.03)
            ax.bar(x, r[metric], W, color=color, zorder=3)
            ax.errorbar(x, r[metric], yerr=[[r[metric] - r[f"{metric}_ci_low"]], [r[f"{metric}_ci_high"] - r[metric]]],
                        color=INK, capsize=4, lw=1.5, capthick=1.5, zorder=4)
            ax.text(x, 0.03, f"{r[metric]:.2f}", ha="center", va="bottom", color="white",
                    fontsize=14, fontweight="bold", zorder=5)
        if metric == "accuracy":
            maj = stats[stats["shape"] == shape].majority_rate.iloc[0]
            ax.plot([gi - 0.47, gi + 0.47], [maj, maj], ls=(0, (4, 3)), color=INK2, lw=1.6, zorder=2)
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
handles += [Line2D([0], [0], color=INK, lw=1.5, marker="_", markersize=10, label="95% bootstrap CI (new, not in report)"),
            Line2D([0], [0], color=INK2, lw=1.6, ls=(0, (4, 3)), label="majority-class accuracy")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.935), ncol=2, frameon=False, columnspacing=1.8)
fig.suptitle("Zero-shot LLM vs NLI-feature classifiers, 91 real examples", x=0.065, y=0.985, ha="left",
             fontsize=19, fontweight="bold")
p = stats[stats.model == "llm"].set_index("shape")
footer(fig, [
    "XGBoost: real-only runs, 5-fold stratified CV, seed 42 (NLI-only predictions recomputed with src/models/ablation.py, matching its saved CSV). LLM: single run, prompt not archived.",
    f"LLM vs NLI-only, McNemar exact p: motte-and-bailey {p.loc['motte_bailey', 'mcnemar_p_llm_vs_nli']:.2f}, strawman {p.loc['strawman', 'mcnemar_p_llm_vs_nli']:.2f}.",
    "No real + synthetic version: the LLM was never run on the 20 synthetic examples (excluded in the original run; mid-report sec 5).",
])
save(fig, "06_llm_vs_nli_real.png")
