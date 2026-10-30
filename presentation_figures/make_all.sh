#!/usr/bin/env bash
# Regenerate every presentation figure from the CSVs in data/annotations/.
# Run from anywhere:  bash presentation_figures/make_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

# Step 0: real-only XGBoost out-of-fold predictions + real-only ablation, using the EXISTING script
# (unchanged). Writes data/annotations/dataset_consolidated_{ablation_results,xgboost_oof_predictions}.csv
# and does not touch any other file.
$PY src/models/ablation.py dataset_consolidated

for f in 01_model_comparison 02_ablation_two_panel 03_explicit_denial_rates 04_error_overlap 05_confusion_matrices 06_llm_vs_nli_real 07_llm_rerun_comparison; do
  $PY presentation_figures/$f.py
done
