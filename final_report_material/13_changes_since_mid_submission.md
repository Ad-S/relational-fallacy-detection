# 13. Changes since the mid-term report

## Data
| | Mid-term | Final |
|---|---|---|
| Real labeled pairs | 91 | 188 |
| Hand-written | 20 | 20 |
| Generated variants | - | 475 |
| Total | 111 | 683 |

The 91 mid-term real pairs are unchanged inside the final data (labels and text verified).

## Protocol
- Mid-term: stratified 5-fold CV, one seed, no grouping, no confidence intervals.
- Final: grouped 5-fold CV (by conversation; variants stay with their source), 3-5 seeds, identical folds for
  all trained models, cluster-bootstrap confidence intervals, ROC-AUC added.

## Models and analyses added
Fine-tuned DeBERTa cross-encoders (NLI-initialised, plain, B-only), trigger-phrase masking, learning curve,
learning-rate sweep, layer-wise probing, open-weight zero-shot LLMs (Qwen2.5 7B/32B), the two-stage
extract-then-verify pipeline, the originals-only experiment, counterfactual flip pairs.

## Zero-shot baseline replaced
- Mid-term: Claude Sonnet 5 zero-shot on the 91 real pairs (accuracy 0.773 MB / 0.809 SM). The same model had
  extracted the claims, and no prompt or raw output was kept, so the result was neither independent nor
  reproducible.
- Final: scripted, deterministic Qwen2.5 runs with saved prompts on all 683 pairs, and an isolated Claude
  Opus 5.5 re-run on the 111 mid-term originals. On the 91 real pairs the re-run reached 0.659 (MB) and 0.787
  (SM) accuracy against the gold labels (0.725 pooled over both tasks, vs 0.791 for the mid-term Sonnet 5 run),
  so the mid-term motte-and-bailey number was not reproduced.
- Agreement **between the two LLM baselines' predictions** (Opus 5.5 re-run vs the mid-term Sonnet 5 zero-shot
  predictions - not gold labels) on the same 91 real pairs, both tasks pooled: 71.4% (Cohen's kappa 0.498).
  The two zero-shot runs disagree on almost a third of the pairs.

## Corrections to mid-term statements
- The mid-term error-overlap counts (48 / 9 / 29 / 10) used XGBoost predictions from the real+synthetic run;
  with the real-only run reported in the mid-term results table they are 45 / 8 / 26 / 11.
- The mid-term feature-ablation headline (NLI-only 0.724 vs full 0.569 accuracy on motte-and-bailey) came from
  real+synthetic data; on real data alone the gap was small (0.659 vs 0.636).
- The lexical features' substring matching was identified as a source of noise (section 5.2).
- The suspected trigger-phrase shortcut was tested: confirmed for motte-and-bailey, not for strawman.

## Mid-term findings that still hold
NLI features are more useful than lexical features; NLI entailment asymmetry alone does not separate the
classes; the trigger-phrase selection shapes the data; zero-shot LLMs are competitive on strawman.
