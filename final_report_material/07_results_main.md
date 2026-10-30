# 7. Main results

Source: `tables/table1_main.csv`, `tables/table2_paired_differences.csv`, `tables/RESULTS.md`.
Grouped 5-fold CV on all 683 pairs; mean +/- standard deviation over seeds (5 seeds for XGBoost and the
main DeBERTa-NLI model, 3 for the others). MB = motte-and-bailey (n=353), SM = strawman (n=330).

## 7.1 Main comparison (Figure: `fig1_roc`)

| Model | Task | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Majority class | MB | 0.640 | - | 0 | 0.000 | 0.500 |
| XGBoost, lexical only | MB | 0.550 +/- 0.005 | 0.219 | 0.098 | 0.135 +/- 0.011 | 0.388 +/- 0.013 |
| XGBoost, NLI only | MB | 0.666 +/- 0.011 | 0.548 | 0.409 | 0.468 +/- 0.028 | 0.652 +/- 0.012 |
| XGBoost, NLI + lexical | MB | 0.638 +/- 0.010 | 0.496 | 0.353 | 0.412 +/- 0.015 | 0.592 +/- 0.010 |
| DeBERTa plain (A,C,B) | MB | 0.646 +/- 0.021 | 0.508 | 0.559 | 0.531 +/- 0.027 | 0.629 +/- 0.022 |
| **DeBERTa-NLI (A,C,B)** | MB | **0.665 +/- 0.030** | 0.537 | 0.540 | **0.538 +/- 0.029** | **0.674 +/- 0.013** |
| DeBERTa-NLI (B only) | MB | 0.587 +/- 0.004 | 0.446 | 0.606 | 0.514 +/- 0.005 | 0.616 +/- 0.011 |
| Majority class | SM | 0.545 | 0.545 | 1 | 0.706 | 0.500 |
| XGBoost, lexical only | SM | 0.561 +/- 0.010 | 0.585 | 0.674 | 0.626 +/- 0.009 | 0.572 +/- 0.015 |
| XGBoost, NLI only | SM | 0.673 +/- 0.006 | 0.678 | 0.761 | 0.717 +/- 0.007 | 0.698 +/- 0.010 |
| XGBoost, NLI + lexical | SM | 0.682 +/- 0.016 | 0.688 | 0.767 | 0.725 +/- 0.012 | 0.701 +/- 0.015 |
| DeBERTa plain (A,B) | SM | 0.652 +/- 0.023 | 0.670 | 0.719 | 0.692 +/- 0.012 | 0.699 +/- 0.035 |
| **DeBERTa-NLI (A,B)** | SM | **0.701 +/- 0.015** | 0.711 | 0.762 | **0.736 +/- 0.013** | **0.770 +/- 0.014** |
| DeBERTa-NLI (B only) | SM | 0.655 +/- 0.029 | 0.666 | 0.735 | 0.699 +/- 0.023 | 0.704 +/- 0.041 |

(For the majority row on strawman the majority class is the fallacy, so "majority" = "always strawman".)
Zero-shot LLM rows are in `09_llm_baselines_and_two_stage.md`.

## 7.2 Paired differences (accuracy, cluster bootstrap 95% CI, seeds 0-2)

| Comparison | Task | Difference | 95% CI | Significant? |
|---|---|---|---|---|
| Fine-tuned DeBERTa-NLI - XGBoost NLI+lexical | MB | +0.009 | [-0.063, +0.083] | no |
| Fine-tuned DeBERTa-NLI - XGBoost NLI+lexical | SM | +0.007 | [-0.062, +0.077] | no |
| NLI initialisation - plain initialisation | MB | +0.005 | [-0.050, +0.061] | no |
| NLI initialisation - plain initialisation | SM | +0.045 | [-0.005, +0.097] | borderline |
| Full exchange - B only | MB | **+0.063** | **[+0.007, +0.116]** | **yes** |
| Full exchange - B only | SM | +0.042 | [-0.021, +0.100] | no |

## 7.3 What to conclude
1. **Both tasks are learnable above chance**, strawman more clearly than motte-and-bailey. Best ROC-AUC:
   0.770 (strawman) and 0.674 (motte-and-bailey), both from fine-tuned DeBERTa-NLI.
2. **Ranking quality improves with fine-tuning** (AUC +0.07 for strawman, +0.08 for motte-and-bailey over
   XGBoost NLI+lexical), but **accuracy at threshold 0.5 does not significantly improve** - the gains are in
   how examples are ordered, not in the default decisions.
3. **Motte-and-bailey is hard:** accuracy of every model is close to the majority rate (0.640), and F1 is
   close to the "always fallacy" value (0.529). AUC is the meaningful number here.
4. **The lexical features hurt or do nothing:** lexical-only is below chance on motte-and-bailey (AUC 0.388),
   and adding them to NLI features lowers motte-and-bailey AUC (0.652 -> 0.592). NLI features carry the
   feature model.
5. **NLI pretraining matters for strawman only** (AUC 0.770 vs 0.699; accuracy +0.045, borderline). A strawman
   is an attribution that A does not support - an inference relation; a motte-and-bailey retreat is not a
   clean entailment pattern.
6. **The full exchange matters most for motte-and-bailey**: the only clearly significant difference in the
   study (+0.063 accuracy over B alone). Seeing A and the challenge C, not just the reply, is needed to
   recognise a retreat.
