"""
Feature ablation: NLI-only vs specificity-only vs NLI+specificity, per shape.
Also saves out-of-fold predictions for the full-feature model (used later
for error analysis) -- cross-validation naturally gives us a prediction for
every example without needing a separate held-out set.

Usage: python3 src/models/ablation.py [dataset_stem]  (default: dataset_final)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ANNOT_DIR = Path("data/annotations")
N_FOLDS = 5
SEED = 42
DATASET_STEM = sys.argv[1] if len(sys.argv) > 1 else "dataset_final"

NLI_FEATURES = ["entail_AB", "neutral_AB", "contra_AB", "entail_BA", "neutral_BA", "contra_BA"]
SPECIFICITY_FEATURES = ["hedge_delta", "hedge_rate_delta", "absolute_delta",
                         "length_ratio_B_over_A", "explicit_denial_B"]

FEATURE_SETS = {
    "NLI only": NLI_FEATURES,
    "Specificity only": SPECIFICITY_FEATURES,
    "NLI + Specificity": NLI_FEATURES + SPECIFICITY_FEATURES,
}


def load_joined():
    base = pd.read_csv(ANNOT_DIR / f"{DATASET_STEM}.csv")
    nli = pd.read_csv(ANNOT_DIR / f"{DATASET_STEM}_nli_features.csv")
    spec = pd.read_csv(ANNOT_DIR / f"{DATASET_STEM}_specificity_features.csv")
    return base.merge(nli, on="pair_id", how="inner").merge(spec, on="pair_id", how="inner")


def run_one(sub, feature_cols, pair_ids, save_oof=False):
    X = sub[feature_cols].values
    y = sub["y"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_pred, all_ids = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                             random_state=SEED, eval_metric="logloss")
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        all_true.extend(y[test_idx].tolist())
        all_pred.extend(preds.tolist())
        all_ids.extend(pair_ids.iloc[test_idx].tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="binary", zero_division=0
    )
    acc = np.mean(np.array(all_true) == np.array(all_pred))

    oof = None
    if save_oof:
        oof = pd.DataFrame({"pair_id": all_ids, "true_y": all_true, "pred_y": all_pred})

    return {"accuracy": round(acc, 3), "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3)}, oof


def main():
    df = load_joined()
    results = []
    oof_frames = []

    for shape_name, positive_label in [("motte_bailey", "MOTTE_BAILEY"), ("strawman", "STRAWMAN")]:
        sub = df[df["type_candidate"] == shape_name].copy()
        sub["y"] = (sub["label"] == positive_label).astype(int)

        for set_name, cols in FEATURE_SETS.items():
            is_full = set_name == "NLI + Specificity"
            metrics, oof = run_one(sub, cols, sub["pair_id"], save_oof=is_full)
            metrics.update({"shape": shape_name, "feature_set": set_name, "n": len(sub)})
            results.append(metrics)
            print(f"{shape_name:14s} | {set_name:18s} | n={len(sub)} | "
                  f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}")
            if is_full:
                oof["shape"] = shape_name
                oof["positive_label"] = positive_label
                oof_frames.append(oof)

    out_path = ANNOT_DIR / f"{DATASET_STEM}_ablation_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shape", "feature_set", "n", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote ablation table to {out_path}")

    oof_all = pd.concat(oof_frames, ignore_index=True)
    oof_path = ANNOT_DIR / f"{DATASET_STEM}_xgboost_oof_predictions.csv"
    oof_all.to_csv(oof_path, index=False)
    print(f"Wrote out-of-fold predictions to {oof_path}")


if __name__ == "__main__":
    main()
