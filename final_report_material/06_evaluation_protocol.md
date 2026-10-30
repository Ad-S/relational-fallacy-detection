# 6. Evaluation protocol

## 6.1 Cross-validation
- 5-fold **StratifiedGroupKFold**, stratified by label and grouped by `conversation_id`, shuffled with the
  seed of the run.
- An example, its generated variants, and other pairs from the same thread always share a fold.
- **All trained models use identical fold assignments** (XGBoost copies the folds recorded by the
  fine-tuning run; all fine-tuning configurations were verified to have the same folds).
- Every one of the 683 pairs is predicted exactly once per seed, by a model that never saw it or any of
  its variants. Metrics are computed on the pooled held-out predictions of one seed, then averaged over
  seeds (mean +/- standard deviation).

## 6.2 Metrics
- Accuracy, precision, recall and F1 for the positive (fallacy) class.
- **ROC-AUC** from predicted probabilities: threshold-free and the most informative metric for
  motte-and-bailey, whose classes are imbalanced (36% positive).
- Zero-shot LLMs output labels only, so they have no AUC; they appear as single points in ROC space.

## 6.3 Reference baselines
| Task | Majority-class accuracy | "Always fallacy" F1 |
|---|---|---|
| Motte-and-bailey (n=353) | 0.640 | 0.529 |
| Strawman (n=330) | 0.545 | 0.706 |

Report F1 next to the "always fallacy" value: a model that labels everything a fallacy already reaches F1
0.529 / 0.706. Qwen2.5-7B zero-shot does exactly this.

## 6.4 Confidence intervals and paired comparisons
- Paired differences in accuracy between two models, using per-example correctness averaged over seeds 0-2.
- **Cluster bootstrap**: 2,000 resamples of conversation groups (97 for motte-and-bailey, 101 for
  strawman), so correlated examples (an example and its variants) are resampled together.
- A difference is called significant only when its 95% interval excludes 0.

## 6.5 Subsets reported separately
- By data source: real CMV (188), hand-written (20), generated variants (475).
- The 208 originals (real + hand-written), for the "do variants help" experiment and for the comparison
  with Claude Opus 5.5 on the 111 mid-term originals.
