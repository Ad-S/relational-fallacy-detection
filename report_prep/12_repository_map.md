# 12. Repository Map

Repo: https://github.com/Ad-S/relational-fallacy-detection (public)

## Source code (`src/`)

### `src/extraction/` — Stage 1-2 of the data pipeline
- `build_candidates.py` — walks the ConvoKit CMV corpus's reply trees,
  extracts motte-bailey-shaped and strawman-shaped candidates by structure
  alone. Outputs the raw (unfiltered) candidate pools — too large for git
  (278MB/437MB), regenerate by rerunning this script against the downloaded
  corpus.
- `filter_candidates.py` — applies the keyword/trigger-phrase filter to the
  raw candidate pools. Outputs `data/processed/candidates_motte_filtered.jsonl`
  (3,109 rows) and `data/processed/candidates_strawman_filtered.jsonl`
  (1,969 rows) — both committed to the repo.

### `src/annotation/` — batch sampling and dataset consolidation
- `prepare_batch.py` — samples the next batch of 50 candidates (25
  motte-bailey + 25 strawman) from the filtered pools, without overlap with
  previous batches (tracked via `data/annotations/used_pair_keys.json`),
  cleans Reddit markup. Takes a batch number argument.
- `merge_for_labeling.py` — merges a batch's raw CSV with its extracted-claims
  CSV into one `batch_N_to_label.csv` file with clean claims and a blank
  `label`/`annotator_notes` column, ready for hand-labeling.
- `consolidate_dataset.py` — merges the hand-labeled `annotated_batch_{1,2,3}
  _to_label.csv` files into one clean dataset, normalizes labels, drops
  `UNCERTAIN`/blank rows. Outputs `data/annotations/dataset_consolidated.csv`
  (91 real examples).

### `src/features/` — Stage 2 of the methodology
- `nli_features.py` — runs the frozen DeBERTa-v3 NLI model on each claim
  pair, both directions, outputs 6 features per pair.
- `specificity_features.py` — computes the 5 lexicon-based specificity
  features per pair.

### `src/models/` — classifiers
- `train_xgboost.py` — main XGBoost training/eval script, 5-fold stratified
  CV, per shape. Takes a dataset name argument (e.g. `dataset_consolidated`
  or `dataset_final`).
- `ablation.py` — feature-ablation variant (NLI-only / specificity-only /
  combined), also saves out-of-fold predictions for error analysis.

### `src/figures/` — all plotting code
- `make_figures.py` — `model_comparison.png`, `dataset_funnel.png`,
  `label_distribution.png`.
- `make_nli_figures.py` — `nli_scatter.png`, `nli_asymmetry_by_label.png`.
- `make_synthetic_comparison.py` — `real_vs_synthetic_features.png`.

## Data (`data/`)

### `data/raw/` — the downloaded ConvoKit corpus (excluded from git via
`.gitignore` due to size; reproducible via `convokit.download(...)`, see
file 02)

### `data/processed/` — filtered candidate pools (committed)
- `candidates_motte_filtered.jsonl` (3,109 rows)
- `candidates_strawman_filtered.jsonl` (1,969 rows)

### `data/annotations/` — everything annotation-related
- `batch_{1..6}.csv` — raw sampled candidates per batch (50 rows each)
- `batch_{1..6}_extracted.csv` — claim-extraction output per batch (status +
  extracted claims + notes)
- `batch_{1..6}_to_label.csv` — merged, ready-to-label files (batches 4-6
  exist with claims extracted but `label` column still blank — this is
  exactly the "149 additional usable candidates, not yet hand-labeled" state
  described in files 03/10)
- `annotated_batch_{1,2,3}_to_label.csv` — the actual hand-labeled files (the
  only batches labeled so far)
- `ANNOTATION_GUIDELINE.md` — the annotation guideline with worked examples
- `dataset_consolidated.csv` — 91 real examples (from batches 1-3 only)
- `dataset_consolidated_nli_features.csv`, `dataset_consolidated_
  specificity_features.csv` — features computed on the 91-real set
- `xgboost_results_summary.csv` — XGBoost results on the 91-real set
- `synthetic_examples.csv` — the 20 hand-written synthetic examples
- `dataset_final.csv` — 111 examples (91 real + 20 synthetic) — **the
  primary dataset file for most analyses**
- `dataset_final_nli_features.csv`, `dataset_final_specificity_features.csv`
  — features computed on the 111-example set
- `dataset_final_xgboost_results_summary.csv` — XGBoost results on 111
- `dataset_final_ablation_results.csv` — feature ablation results
- `dataset_final_xgboost_oof_predictions.csv` — XGBoost per-example
  out-of-fold predictions (used for error analysis)
- `zeroshot_baseline_predictions.csv` — zero-shot LLM per-example predictions
- `ERROR_ANALYSIS.md` — full error analysis writeup
- `used_pair_keys.json` — tracks which candidates have been sampled across
  batches, to prevent overlap

## Figures (`figures/`)

All 6 figures described in detail in file 11.

## Top-level documentation
- `REPORT_CONTENT_DOSSIER.md` — an earlier, alternative version of this same
  material organized directly against a 15-section ACL paper structure
  (redundant with this `report_prep/` folder's content; use whichever
  organization you find easier to write from, not both)
- `report_prep/` — this folder

## What does not exist yet (do not cite as if it does)
- No `README.md` at the repo root yet.
- No LaTeX source for the report itself.
- No fine-tuned transformer model or training code.
- No trigger-phrase-masking ablation code or results.
- Batches 4-6 have no `annotated_batch_N_to_label.csv` files (not yet hand-labeled).
