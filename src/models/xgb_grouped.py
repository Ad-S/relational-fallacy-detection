"""XGBoost baseline on the 683-row dataset with the SAME grouped folds as finetune/train_cv.py.

Folds: taken directly from the fine-tuning run's recorded assignments (finetune/session_results/nli_pair/
predictions.csv stores seed and fold per example), so XGBoost and the fine-tuned models are evaluated on
exactly the same grouped folds (recomputing StratifiedGroupKFold can differ across scikit-learn versions). XGBoost settings as in the mid-report
(50 trees, depth 3, lr 0.1, seed 42). Feature sets: NLI only, specificity only, NLI + specificity.

Usage: python3 src/models/xgb_grouped.py
Writes: finetune/session_results/xgb_grouped/{predictions.csv, summary.csv}
"""
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "annotations" / "dataset_augmented_all_v2.csv"
NLI = ROOT / "finetune" / "session_results" / "features" / "dataset_augmented_all_v2_nli_features.csv"
SPEC = ROOT / "data" / "annotations" / "dataset_augmented_all_v2_specificity_features.csv"
OUT = ROOT / "finetune" / "session_results" / "xgb_grouped"
FOLDS = ROOT / "finetune" / "session_results" / "nli_pair" / "predictions.csv"
POSITIVE = {"motte_bailey": "MOTTE_BAILEY", "strawman": "STRAWMAN"}
NLI_F = ["entail_AB", "neutral_AB", "contra_AB", "entail_BA", "neutral_BA", "contra_BA"]
SPEC_F = ["hedge_delta", "hedge_rate_delta", "absolute_delta", "length_ratio_B_over_A", "explicit_denial_B"]
SETS = {"NLI only": NLI_F, "Specificity only": SPEC_F, "NLI + Specificity": NLI_F + SPEC_F}
SEEDS = [0, 1, 2, 3, 4]


def main():
    data = pd.read_csv(DATA, keep_default_na=False)
    df = data.merge(pd.read_csv(NLI), on="pair_id", how="left").merge(pd.read_csv(SPEC), on="pair_id", how="left")
    assert len(df) == len(data) and df[NLI_F + SPEC_F].notna().all().all()
    fold_of = pd.read_csv(FOLDS).set_index(["shape", "seed", "pair_id"]).fold
    preds = []
    for shape, pos in POSITIVE.items():
        sub = df[df.type_candidate == shape].reset_index(drop=True)
        y = (sub.label == pos).astype(int).values
        for seed in SEEDS:
            f = np.array([fold_of[(shape, seed, pid)] for pid in sub.pair_id])
            folds = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]
            for name, cols in SETS.items():
                X = sub[cols].values
                for fold, (tr, te) in enumerate(folds):
                    clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42,
                                        eval_metric="logloss").fit(X[tr], y[tr])
                    prob = clf.predict_proba(X[te])[:, 1]
                    for i, p in zip(te, prob):
                        preds.append(dict(config=f"xgb_{name}", shape=shape, seed=seed, fold=fold, pair_id=sub.pair_id[i],
                                          source=sub.source[i], y=int(y[i]), prob=float(p), pred=int(p > 0.5)))
    preds = pd.DataFrame(preds)
    OUT.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT / "predictions.csv", index=False)
    rows = []
    for (cfg, shape, seed), g in preds.groupby(["config", "shape", "seed"]):
        tp = ((g.y == 1) & (g.pred == 1)).sum(); fp = ((g.y == 0) & (g.pred == 1)).sum(); fn = ((g.y == 1) & (g.pred == 0)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(dict(config=cfg, shape=shape, seed=seed, n=len(g), accuracy=(g.y == g.pred).mean(),
                         precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False)
    print(summ.groupby(["config", "shape"])[["accuracy", "f1"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
