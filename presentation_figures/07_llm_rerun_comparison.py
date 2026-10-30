"""Fig 7: NEW scripted zero-shot LLM run vs NLI-only and full XGBoost, real-only (n=91) and real + synthetic (n=111).

LLM: presentation_figures/llm_rerun/zeroshot_rerun_predictions.csv, produced by
     llm_rerun/run_zeroshot_llm.py (Claude Opus 5.5 via headless Claude Code, fresh isolated session per
     example, saved prompt, raw outputs in llm_rerun/raw/). This is a NEW baseline, not the mid-report's.
XGBoost full model: dataset_consolidated_xgboost_oof_predictions.csv (real) and
     dataset_final_xgboost_oof_predictions.csv (real + synthetic).
XGBoost NLI-only: recomputed with src/models/ablation.py's run_one(), asserted equal to the saved
     ablation CSVs.
Whiskers: paired bootstrap 95% CI (10,000 resamples, seed 0).

Produces 07_llm_rerun_comparison_accuracy.png and 07_llm_rerun_comparison_f1.png.
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
MODEL_NAME = "Claude Opus 5.5"
MODELS = [("nli", "XGBoost, NLI features only", AQUA),
          ("xgb", "XGBoost, NLI + specificity (full)", C_XGB),
          ("llm", f"Zero-shot LLM, new scripted run ({MODEL_NAME})", C_LLM)]
PANELS = [("dataset_consolidated", "Real only", "XGBoost trained on real only"),
          ("dataset_final", "Real + synthetic", "XGBoost trained on real + synthetic")]

style()
llm = pd.read_csv(OUT / "llm_rerun" / "zeroshot_rerun_predictions.csv")
missing = llm[llm.predicted_label.isna() | (llm.predicted_label == "")]
assert missing.empty, f"LLM predictions missing for {list(missing.pair_id)} - rerun run_zeroshot_llm.py"
assert set(llm.models_reported) == {"claude-opus-5-5"}, set(llm.models_reported)


def panel_frame(stem):
    gold = pd.read_csv(ANNOT / f"{stem}.csv")
    feats = gold.merge(pd.read_csv(ANNOT / f"{stem}_nli_features.csv"), on="pair_id") \
                .merge(pd.read_csv(ANNOT / f"{stem}_specificity_features.csv"), on="pair_id")
    saved_abl = pd.read_csv(ANNOT / f"{stem}_ablation_results.csv")
    full = pd.read_csv(ANNOT / f"{stem}_xgboost_oof_predictions.csv")[["pair_id", "true_y", "pred_y"]]
    nli = {}
    for shape, pos, _ in SHAPES:
        sub = feats[feats.type_candidate == shape].copy()
        sub["y"] = (sub.label == pos).astype(int)
        m, oof = ablation.run_one(sub, ablation.NLI_FEATURES, sub["pair_id"], save_oof=True)
        ref = saved_abl[(saved_abl["shape"] == shape) & (saved_abl.feature_set == "NLI only")].iloc[0]
        for col in ("accuracy", "precision", "recall", "f1"):
            assert round(m[col], 3) == round(ref[col], 3), (stem, shape, col)
        nli.update(dict(zip(oof.pair_id, oof.pred_y)))
    df = gold[["pair_id", "type_candidate", "label"]].merge(full, on="pair_id") \
             .merge(llm[["pair_id", "predicted_label"]], on="pair_id")
    assert len(df) == len(gold), (stem, len(df), len(gold))
    df["gold"] = (df.label == df.type_candidate.map(POS_LABEL)).astype(int)
    assert (df.gold == df.true_y).all()
    df["xgb"] = df.pred_y.astype(int)
    df["nli"] = df.pair_id.map(nli).astype(int)
    df["llm"] = (df.predicted_label == df.type_candidate.map(POS_LABEL)).astype(int)
    return df


rng = np.random.default_rng(SEED)
rows = []
frames = {}
for stem, title, _ in PANELS:
    df = frames[stem] = panel_frame(stem)
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
            r = dict(dataset=stem, shape=shape, model=key, n=n, majority_rate=majority_rate(y),
                     accuracy=m["accuracy"], precision=m["precision"], recall=m["recall"], f1=m["f1"],
                     tn=m["tn"], fp=m["fp"], fn=m["fn"], tp=m["tp"],
                     accuracy_ci_low=np.percentile(acc_b, 2.5), accuracy_ci_high=np.percentile(acc_b, 97.5),
                     f1_ci_low=np.percentile(f1_b, 2.5), f1_ci_high=np.percentile(f1_b, 97.5))
            if key == "llm":
                for other in ("nli", "xgb"):
                    b = int(((sub.llm == y) & (sub[other] != y)).sum())
                    c = int(((sub.llm != y) & (sub[other] == y)).sum())
                    r[f"mcnemar_p_vs_{other}"] = mcnemar_exact(b, c)
            rows.append(r)
stats = pd.DataFrame(rows)
stats.to_csv(DER / "llm_rerun_comparison_stats.csv", index=False)

# synthetic-only accuracy of the LLM, and agreement with the ORIGINAL (mid-report) LLM labels on the 91 real
allf = frames["dataset_final"]
syn_ids = set(pd.read_csv(ANNOT / "dataset_final.csv").query("source != 'reddit_cmv'").pair_id)
syn = allf[allf.pair_id.isin(syn_ids)]
orig = pd.read_csv(ANNOT / "zeroshot_baseline_predictions.csv")[["pair_id", "predicted_label"]] \
    .rename(columns={"predicted_label": "original_llm"})
agree = llm.merge(orig, on="pair_id")
po = (agree.predicted_label == agree.original_llm).mean()
cats = sorted(set(agree.predicted_label) | set(agree.original_llm))
pe = sum((agree.predicted_label == c).mean() * (agree.original_llm == c).mean() for c in cats)
kappa = (po - pe) / (1 - pe)
extra = pd.DataFrame([dict(item="llm_rerun_accuracy_synthetic_only", n=len(syn), value=(syn.llm == syn.gold).mean()),
                      dict(item="agreement_with_original_llm_real91", n=len(agree), value=po),
                      dict(item="cohen_kappa_with_original_llm_real91", n=len(agree), value=kappa)])
extra.to_csv(DER / "llm_rerun_extra.csv", index=False)
print(extra.round(3).to_string(index=False))


def draw(metric, fname, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.69, bottom=0.27, wspace=0.07)
    W = 0.25
    for ax, (stem, title, tag) in zip(axes, PANELS):
        for gi, (shape, pos, name) in enumerate(SHAPES):
            for mi, (key, _, color) in enumerate(MODELS):
                r = stats[(stats.dataset == stem) & (stats["shape"] == shape) & (stats.model == key)].iloc[0]
                x = gi + (mi - 1) * (W + 0.03)
                ax.bar(x, r[metric], W, color=color, zorder=3)
                ax.errorbar(x, r[metric], yerr=[[r[metric] - r[f"{metric}_ci_low"]], [r[f"{metric}_ci_high"] - r[metric]]],
                            color=INK, capsize=4, lw=1.5, capthick=1.5, zorder=4)
                ax.text(x, 0.03, f"{r[metric]:.2f}", ha="center", va="bottom", color="white",
                        fontsize=14, fontweight="bold", zorder=5)
            if metric == "accuracy":
                maj = stats[(stats.dataset == stem) & (stats["shape"] == shape)].majority_rate.iloc[0]
                ax.plot([gi - 0.47, gi + 0.47], [maj, maj], ls=(0, (4, 3)), color=INK2, lw=1.6, zorder=2)
        sel = stats[stats.dataset == stem]
        ax.set_xticks(range(len(SHAPES)))
        ax.set_xticklabels([f"{name}\nn={int(sel[sel['shape'] == s].n.iloc[0])}, majority {sel[sel['shape'] == s].majority_rate.iloc[0]:.2f}"
                            for s, _, name in SHAPES])
        ax.set_xlim(-0.6, len(SHAPES) - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, loc="left", pad=26)
        ax.text(0, 1.035, tag, transform=ax.transAxes, fontsize=12, color=INK2, va="bottom")
        clean_axis(ax)
    axes[0].set_ylabel(ylabel)
    axes[1].tick_params(labelleft=False)
    handles = [Patch(color=c, label=l) for _, l, c in MODELS]
    handles.append(Line2D([0], [0], color=INK, lw=1.5, marker="_", markersize=10, label="95% paired bootstrap CI"))
    if metric == "accuracy":
        handles.append(Line2D([0], [0], color=INK2, lw=1.6, ls=(0, (4, 3)), label="majority-class accuracy"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.935), ncol=2, frameon=False, columnspacing=1.8)
    fig.suptitle(f"New zero-shot LLM run vs NLI-feature classifiers: {ylabel}", x=0.065, y=0.985, ha="left",
                 fontsize=19, fontweight="bold")
    e = extra.set_index("item").value
    footer(fig, [
        f"LLM: NEW scripted run, not the mid-report baseline. {MODEL_NAME}, one fresh tool-less session per example, saved prompt (definitions only, no dataset examples), single run, no training.",
        f"Same LLM predictions in both panels (it is not trained); only the evaluation set changes. LLM accuracy on the 20 synthetic examples alone: {e['llm_rerun_accuracy_synthetic_only']:.2f}.",
        f"Agreement with the original mid-report LLM labels on the 91 real: {e['agreement_with_original_llm_real91']:.0%} (Cohen's kappa {e['cohen_kappa_with_original_llm_real91']:.2f}). XGBoost: 5-fold stratified CV, seed 42.",
    ])
    save(fig, fname)


draw("accuracy", "07_llm_rerun_comparison_accuracy.png", "Accuracy")
draw("f1", "07_llm_rerun_comparison_f1.png", "F1 (positive class)")
