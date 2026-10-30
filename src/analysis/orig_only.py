"""Do the generated variants help on the original examples?

Compares, on the 208 original examples (188 real + 20 hand-written):
  (a) DeBERTa-NLI fine-tuned on originals only (nli_pair_orig_only run),
  (b) XGBoost NLI+spec trained on originals only, on exactly the same folds as (a),
  (c) DeBERTa-NLI fine-tuned on all 683 rows (nli_pair run), scored on the originals of its test folds.
(a)/(b) share folds; (c) uses the 683-row folds, so (c) vs (a) is an unpaired comparison (seed spread reported).

Usage: python3 src/analysis/orig_only.py   (run after final_results.py and llm_results.py)
Writes: results_final/table11_originals_only.csv; appends to RESULTS.md
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import final_results as F

NLI_F = ["entail_AB", "neutral_AB", "contra_AB", "entail_BA", "neutral_BA", "contra_BA"]
SPEC_F = ["hedge_delta", "hedge_rate_delta", "absolute_delta", "length_ratio_B_over_A", "explicit_denial_B"]


def metrics(g):
    y, pr, prob = g.y.values, g.pred.values, g.prob.values
    tp, fp, fn = ((y == 1) & (pr == 1)).sum(), ((y == 0) & (pr == 1)).sum(), ((y == 1) & (pr == 0)).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return pd.Series(dict(accuracy=(y == pr).mean(), f1=2 * p * r / (p + r) if p + r else 0.0,
                          auc=F.roc_auc_score(y, prob)))


def main():
    ft_orig = pd.read_csv(F.RES / "nli_pair_orig_only" / "predictions.csv", keep_default_na=False)
    data = F.data
    feats = data.merge(pd.read_csv(F.RES / "features" / "dataset_augmented_all_v2_nli_features.csv"), on="pair_id") \
                .merge(pd.read_csv(F.ROOT / "data" / "annotations" / "dataset_augmented_all_v2_specificity_features.csv"), on="pair_id") \
                .set_index("pair_id")
    xrows = []
    for (shape, seed), g in ft_orig.groupby(["shape", "seed"]):
        for k in range(5):
            tr, te = g[g.fold != k].pair_id, g[g.fold == k].pair_id
            Xtr, Xte = feats.loc[tr, NLI_F + SPEC_F].values, feats.loc[te, NLI_F + SPEC_F].values
            ytr = g.set_index("pair_id").loc[tr, "y"].values
            clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="logloss").fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]
            yte = g.set_index("pair_id").loc[te, "y"].values
            for pid, y, pb in zip(te, yte, prob):
                xrows.append(dict(shape=shape, seed=seed, fold=k, pair_id=pid, y=int(y), prob=float(pb), pred=int(pb > 0.5)))
    xgb_orig = pd.DataFrame(xrows)

    all_runs = F.load()
    orig_ids = set(data[data.source.isin(["reddit_cmv", "handwritten"])].pair_id)
    ft_all = all_runs[(all_runs.config == "nli_pair") & all_runs.pair_id.isin(orig_ids) & all_runs.seed.isin([0, 1, 2])]

    rows, lines = [], ["", "## Do the generated variants help on the original examples? (208 originals; seeds 0-2)", "",
                       "| Training data | Model | Shape | n | Accuracy | F1 | AUC |", "|---|---|---|---|---|---|---|"]
    for label, model, df in [("originals only", "XGBoost NLI+spec", xgb_orig), ("originals only", "DeBERTa-NLI", ft_orig),
                             ("originals + 475 variants", "DeBERTa-NLI", ft_all)]:
        for shape in F.SHAPES:
            s = df[df["shape"] == shape].groupby("seed").apply(metrics, include_groups=False)
            n = df[df["shape"] == shape].pair_id.nunique()
            rows.append(dict(training=label, model=model, shape=shape, n=n, **{f"{m}_mean": s[m].mean() for m in s},
                             **{f"{m}_std": s[m].std(ddof=0) for m in s}))
            lines.append(f"| {label} | {model} | {F.SHAPES[shape]} | {n} | {s.accuracy.mean():.3f} ± {s.accuracy.std(ddof=0):.3f} | "
                         f"{s.f1.mean():.3f} ± {s.f1.std(ddof=0):.3f} | {s.auc.mean():.3f} ± {s.auc.std(ddof=0):.3f} |")
    for shape in F.SHAPES:
        y = (data[data.pair_id.isin(orig_ids) & (data.type_candidate == shape)].label != "NONE").mean()
        lines.append(f"| - | Majority class | {F.SHAPES[shape]} | | {max(y, 1 - y):.3f} | | 0.500 |")
    pd.DataFrame(rows).to_csv(F.OUT / "table11_originals_only.csv", index=False)
    with open(F.OUT / "RESULTS.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
