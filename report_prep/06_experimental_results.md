# 6. Experimental Results

All numbers in this file were re-verified directly from the CSV result files
in `data/annotations/` on 2026-10-02 (not recalled from memory). Source file
named under each table.

## Table A: XGBoost vs. zero-shot LLM, on 91 real examples

Source: `data/annotations/xgboost_results_summary.csv` and
`data/annotations/zeroshot_baseline_predictions.csv` (recomputed directly).

| Shape | Model | N | Positive | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Motte-bailey | XGBoost (NLI+specificity) | 44 | 20 | 0.636 | 0.611 | 0.550 | 0.579 |
| Motte-bailey | Zero-shot LLM | 44 | 20 | 0.773 | 0.750 | 0.750 | 0.750 |
| Strawman | XGBoost (NLI+specificity) | 47 | 23 | 0.617 | 0.619 | 0.565 | 0.591 |
| Strawman | Zero-shot LLM | 47 | 23 | 0.809 | 0.818 | 0.783 | 0.800 |

**Confusion matrices** (rows = true, cols = predicted; order [NONE, positive class]):

Motte-bailey, XGBoost:
```
[[17  7]
 [ 9 11]]
```
Motte-bailey, zero-shot LLM:
```
[[19  5]
 [ 5 15]]
```
Strawman, XGBoost:
```
[[16  8]
 [10 13]]
```
Strawman, zero-shot LLM:
```
[[20  4]
 [ 5 18]]
```

**Chance/majority-class baseline for comparison** (compute and cite this —
it was not in any saved file, but is derivable directly from the label
counts in file 04): on the 91-real set, motte-bailey majority class is NONE
at 24/44 = 0.545 accuracy; strawman majority class is NONE at 24/47 = 0.511
accuracy. XGBoost clearly beats this baseline; the margin over chance is
modest but real.

## Table B: XGBoost, real-only vs. real+synthetic

Source: `data/annotations/xgboost_results_summary.csv` (91-real) and
`data/annotations/dataset_final_xgboost_results_summary.csv` (111-combined).

| Shape | Data | N | Positive | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Motte-bailey | 91 real | 44 | 20 | 0.636 | 0.611 | 0.550 | 0.579 |
| Motte-bailey | 111 real+synthetic | 58 | 28 | 0.569 | 0.545 | 0.643 | 0.590 |
| Strawman | 91 real | 47 | 23 | 0.617 | 0.619 | 0.565 | 0.591 |
| Strawman | 111 real+synthetic | 53 | 29 | 0.547 | 0.586 | 0.586 | 0.586 |

**Observation**: adding the 20 synthetic examples did not improve XGBoost's
performance on either shape, and mildly reduced accuracy on both (F1 is
roughly flat — up slightly for motte-bailey, down slightly for strawman).
See file 07 for the distributional analysis that offers a plausible, evidence
-based explanation.

## Which examples were used where (important for a methods-accuracy check)

- XGBoost 91-real run: `dataset_consolidated.csv` (91 rows, real only)
- XGBoost 111-combined run: `dataset_final.csv` (111 rows, real+synthetic)
- Feature ablation (file 07): run on `dataset_final.csv` (111 rows)
- Zero-shot LLM: run on the 91 real examples only (never run on the synthetic
  20, since their labels were visible in the same context they were
  generated in — see file 05's limitation note)
- Error analysis (file 08): built from the overlap of XGBoost's out-of-fold
  predictions and zero-shot predictions, which is the 91 real examples (the
  only set both models were evaluated on)

## Interpretation notes (facts to turn into your own sentences)

- The zero-shot LLM outperforms the engineered-feature XGBoost model by a
  substantial margin on both shapes, on the same real data (Table A).
- This should be read alongside the independence caveat in file 05 — the
  magnitude of the LLM's lead cannot be taken as a clean, arms-length
  capability comparison given that caveat.
- The real-vs-synthetic comparison (Table B) shows no benefit and a small
  cost from the synthetic augmentation as currently constructed — this is a
  genuine negative/null result and should be reported as such, not
  downplayed or omitted.
