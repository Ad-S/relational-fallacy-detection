# 8. Analysis: what do the models use?

Source tables: `tables/RESULTS.md` and the CSVs named below.

## 8.1 Input ablation: B only vs the full exchange (Figure: `fig2_what_the_model_uses`, left)
| Task | B only (AUC) | Full exchange (AUC) | Accuracy difference (95% CI) |
|---|---|---|---|
| Motte-and-bailey | 0.616 | 0.674 | +0.063 [+0.007, +0.116] |
| Strawman | 0.704 | 0.770 | +0.042 [-0.021, +0.100] |

Reading: a model that sees only B already does better than chance (B carries cues, including the trigger
phrases), but adding A (and C) improves both tasks - significantly for motte-and-bailey. The task is
partly relational, partly solvable from B alone.

## 8.2 Trigger-phrase masking (Figure: `fig2_what_the_model_uses`, right; `tables/table3_masking.csv`)
Selection phrases deleted from B at test time; accuracy on the affected rows only.

| Model | Task | Affected rows | Accuracy, original | Accuracy, masked | F1 all rows, original -> masked |
|---|---|---|---|---|---|
| DeBERTa-NLI (A,C,B) | MB | 218 | 0.645 | **0.581** | 0.538 -> 0.491 |
| DeBERTa-NLI (A,B) | SM | 141 | 0.671 | 0.670 | 0.736 -> 0.741 |
| DeBERTa plain | MB | 218 | 0.625 | 0.641 | 0.531 -> 0.529 |
| DeBERTa plain | SM | 141 | 0.598 | 0.584 | 0.692 -> 0.697 |
| DeBERTa-NLI (B only) | MB | 218 | 0.560 | 0.590 | 0.514 -> 0.439 |
| DeBERTa-NLI (B only) | SM | 141 | 0.629 | 0.600 | 0.699 -> 0.699 |

Reading: the best motte-and-bailey model loses 6.4 accuracy points when "I never said" / "I'm not saying"
style phrases disappear - it partly relies on them. The strawman model is unaffected: its signal does not
come from "so you're saying" style phrases. Interpret with care: deleting a phrase can also remove
genuine content.

## 8.3 Layer-wise probing of frozen encoders (Figure: `fig3_probing`)
| Encoder (frozen) | Task | Best layer | Best AUC | Layer-0 AUC | Last-layer AUC |
|---|---|---|---|---|---|
| DeBERTa-NLI | MB | 18 | **0.718** | 0.611 | 0.650 |
| DeBERTa plain | MB | 9 | 0.696 | 0.614 | 0.634 |
| DeBERTa-NLI | SM | 24 | **0.736** | 0.601 | 0.736 |
| DeBERTa plain | SM | 22 | 0.684 | 0.600 | 0.638 |

Reading:
- Already the embedding layer gives AUC about 0.60 (lexical content alone separates the classes somewhat).
- The signal grows in the middle and late layers; for strawman it peaks in the last layers of the NLI
  encoder and the NLI encoder clearly beats the plain one there (0.736 vs 0.638 at the last layer) -
  consistent with strawman depending on inference knowledge.
- For motte-and-bailey, a linear probe on frozen layer 18 (AUC 0.718) matches or beats full fine-tuning
  (0.674): with this little data, fine-tuning does not add information beyond what the frozen NLI encoder
  already contains.
- Both encoders show a dip around layers 10-14 on motte-and-bailey.

## 8.4 Learning curve (Figure: `fig4_learning_curve`; `tables/table6_learning_curve.csv`)
DeBERTa-NLI, fraction of training conversations kept (test folds unchanged), seeds 0-2.

| Training data | MB AUC | MB F1 | SM AUC | SM F1 |
|---|---|---|---|---|
| 25% | 0.544 +/- 0.077 | 0.460 | 0.659 +/- 0.062 | 0.696 |
| 50% | 0.578 +/- 0.058 | 0.441 | 0.722 +/- 0.050 | 0.713 |
| 75% | 0.637 +/- 0.032 | 0.511 | 0.735 +/- 0.009 | 0.699 |
| 100% | 0.666 +/- 0.007 | 0.528 | 0.763 +/- 0.011 | 0.734 |

Reading: performance rises steadily and has not levelled off; variance across seeds shrinks with more
data. More (real) data is likely to help, motte-and-bailey most.

## 8.5 Learning-rate sensitivity (`tables/table5_lr_sensitivity.csv`)
| Learning rate | MB AUC | SM AUC |
|---|---|---|
| 1e-5 | 0.643 +/- 0.021 | 0.762 +/- 0.002 |
| 1.5e-5 (used) | 0.666 +/- 0.007 | 0.763 +/- 0.011 |
| 3e-5 | 0.668 +/- 0.003 | 0.779 +/- 0.026 |

Reading: results are stable across a 3x range; conclusions do not hinge on the learning rate.

## 8.6 Accuracy by data source (Figure: `fig5_by_source`; `tables/table4_by_source.csv`)
Both tasks pooled, mean over seeds.

| Model | Real CMV (188) | Hand-written (20) | Generated (475) |
|---|---|---|---|
| XGBoost NLI+lexical | 0.657 | 0.610 | 0.662 |
| DeBERTa-NLI (A,C,B) | 0.627 | 0.680 | 0.705 |
| DeBERTa-NLI (B only) | 0.583 | 0.667 | 0.632 |
| Qwen2.5-32B zero-shot | 0.601 | 0.550 | 0.642 |

Reading: on real CMV pairs, fine-tuned DeBERTa-NLI is **not** better than XGBoost; its advantage comes from
the generated variants. Real exchanges are the hardest part of the data. Must be stated in the report.

## 8.7 Do the generated variants help on original examples? (Figure: `fig7_variants_help`; `tables/table11_originals_only.csv`)
Evaluation on the 208 originals (188 real + 20 hand-written), seeds 0-2.

| Training data | Model | MB accuracy | MB AUC | SM accuracy | SM AUC |
|---|---|---|---|---|---|
| Originals only | XGBoost NLI+lexical | 0.615 | 0.523 +/- 0.048 | 0.609 | 0.609 +/- 0.018 |
| Originals only | DeBERTa-NLI | 0.567 | 0.472 +/- 0.067 | 0.615 | 0.592 +/- 0.067 |
| Originals + 475 variants | DeBERTa-NLI | 0.599 | 0.540 +/- 0.034 | 0.628 | **0.694 +/- 0.014** |
| - | Majority class | 0.663 | 0.500 | 0.538 | 0.500 |

Reading:
- **Variants help strawman on real data:** AUC 0.592 -> 0.694 on the originals, with much smaller seed variance.
- **Motte-and-bailey on originals is close to chance** for every trained model (AUC 0.47-0.54; accuracy below
  majority). The aggregate motte-and-bailey results are driven mostly by generated variants.
- With originals only, fine-tuning has about 83 training pairs per fold and per task - too few for a
  435M-parameter model.
- Caveat: the originals-only run uses its own folds (computed on the 208 rows), so the comparison with the
  full-data run is unpaired; XGBoost and DeBERTa within "originals only" share folds.

## 8.8 Counterfactual flip pairs (`tables/table7_flip_pairs.csv`)
20 hand-written examples and their label-flipped edits (A and C identical, B minimally changed), evaluated
in the same fold; counts over seeds.

| Model | Pair evaluations | Both members correct | Prediction differs between the two |
|---|---|---|---|
| XGBoost NLI+lexical | 100 | 0.30 | 0.50 |
| DeBERTa-NLI (A,C,B) | 100 | 0.51 | 0.55 |
| DeBERTa-NLI (B only) | 60 | 0.52 | 0.60 |

Reading: models do not reliably track a minimal edit that changes the label - about half the time both are
right. The B-only model changes its prediction as often as the full model, so the edits are noticed in B
but not reliably interpreted relative to A. Small sample (20 pairs) and generated flip labels.

## 8.9 Overall answer to the central question
- **Strawman:** the evidence points to a partly relational, inference-based signal - NLI pretraining helps,
  the full pair helps, the signal sits in late NLI layers, and removing trigger phrases changes nothing.
- **Motte-and-bailey:** the models partly rely on surface cues - masking the trigger phrase hurts, the
  originals are near chance, and flip pairs are not handled; yet the full exchange is still clearly better
  than the reply alone, so some relational information is used.
