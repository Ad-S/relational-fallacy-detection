"""
Train and evaluate the lightweight engineered-feature classifier (Section 4
of the proposal): NLI features + specificity features -> XGBoost.

Trained and evaluated SEPARATELY per fallacy shape (motte_bailey vs
strawman), each as its own binary fallacy-vs-NONE task, using stratified
k-fold cross-validation since the dataset is small. We don't do a single
train/test split because with ~45 examples per shape, one split would be
too noisy to trust; cross-validation averages over several splits.

Discourse features (same_speaker, challenge_present) are deliberately
excluded here: they are constant within each shape by construction (all
motte_bailey rows have same_speaker=True/challenge=True, all strawman rows
have both False), so they carry zero signal in a per-shape binary task.
They would only matter for a joint 3-way setup, which we're not using as
the main result (see problem statement / methodology section).

Usage: python3 src/models/train_xgboost.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ANNOT_DIR = Path("data/annotations")
N_FOLDS = 5
SEED = 42
DATASET_STEM = sys.argv[1] if len(sys.argv) > 1 else "dataset_consolidated"

FEATURE_COLS = [
    "entail_AB", "neutral_AB", "contra_AB",
    "entail_BA", "neutral_BA", "contra_BA",
    "hedge_delta", "hedge_rate_delta",
    "absolute_delta", "length_ratio_B_over_A", "explicit_denial_B",
]


def load_joined():
    base = pd.read_csv(ANNOT_DIR / f"{DATASET_STEM}.csv")
    nli = pd.read_csv(ANNOT_DIR / f"{DATASET_STEM}_nli_features.csv")
    spec = pd.read_csv(ANNOT_DIR / f"{DATASET_STEM}_specificity_features.csv")

    df = base.merge(nli, on="pair_id", how="inner").merge(spec, on="pair_id", how="inner")
    return df


def run_shape(df, shape_name, positive_label):
    sub = df[df["type_candidate"] == shape_name].copy()
    sub["y"] = (sub["label"] == positive_label).astype(int)

    X = sub[FEATURE_COLS].values
    y = sub["y"].values

    print(f"\n=== {shape_name} ({positive_label} vs NONE) ===")
    print(f"n={len(sub)}  positive={y.sum()}  negative={(y == 0).sum()}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=SEED, eval_metric="logloss",
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="binary", zero_division=0
    )
    acc = np.mean(np.array(all_true) == np.array(all_pred))
    cm = confusion_matrix(all_true, all_pred)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"Confusion matrix (rows=true, cols=pred, [0=NONE,1={positive_label}]):")
    print(cm)

    return {
        "shape": shape_name, "n": len(sub), "positive": int(y.sum()),
        "accuracy": round(acc, 3), "precision": round(precision, 3),
        "recall": round(recall, 3), "f1": round(f1, 3),
    }


def main():
    df = load_joined()
    print(f"Joined dataset: {len(df)} rows with features")

    results = []
    results.append(run_shape(df, "motte_bailey", "MOTTE_BAILEY"))
    results.append(run_shape(df, "strawman", "STRAWMAN"))

    out_path = ANNOT_DIR / f"{DATASET_STEM}_xgboost_results_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shape", "n", "positive", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
