# 9. Zero-shot LLM baselines and the two-stage pipeline

Sources: `tables/table8_llm_zeroshot.csv`, `table9_two_stage.csv`, `table10_paired_llm.csv`, `RESULTS.md`.
Method details: `05_features_models_methods.md`, sections 5.7-5.8.

## 9.1 Zero-shot on all 683 pairs

| Model | Task | Unparsed | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | MB | 0 | 0.360 | 0.360 | 1.000 | 0.529 |
| Qwen2.5-7B | SM | 0 | 0.555 | 0.550 | 1.000 | 0.710 |
| Qwen2.5-32B | MB | 1 | 0.555 | 0.443 | 0.913 | 0.596 |
| Qwen2.5-32B | SM | 1 | 0.706 | 0.689 | 0.839 | 0.757 |
| (fine-tuned DeBERTa-NLI, for reference) | MB | - | 0.665 | 0.537 | 0.540 | 0.538 |
| (fine-tuned DeBERTa-NLI, for reference) | SM | - | 0.701 | 0.711 | 0.762 | 0.736 |

- **Qwen2.5-7B labels every pair a fallacy** (recall 1.0): its numbers equal the "always fallacy" baseline.
- **Qwen2.5-32B matches the fine-tuned model on strawman** and has higher F1 there, but **over-predicts
  motte-and-bailey** (recall 0.91, about 65% false-positive rate), so its accuracy (0.555) is below the
  majority rate (0.640). In ROC space (`fig1_roc`) its strawman point lies on the fine-tuned curve; its
  motte-and-bailey point lies above the curves but at a high false-positive rate.
- Larger model, much better behaviour: 7B -> 32B changes the strawman accuracy from 0.555 to 0.706.

Fine-tuned DeBERTa-NLI minus zero-shot (accuracy, cluster bootstrap 95% CI):

| Comparison | Task | Difference | 95% CI |
|---|---|---|---|
| vs Qwen2.5-7B | MB | +0.291 | [+0.177, +0.397] |
| vs Qwen2.5-7B | SM | +0.142 | [+0.050, +0.241] |
| vs Qwen2.5-32B | MB | +0.095 | [-0.011, +0.199] |
| vs Qwen2.5-32B | SM | -0.009 | [-0.099, +0.077] |

## 9.2 On the 111 original pairs of the mid-term dataset (91 real + 20 hand-written)

| Model | MB accuracy | MB F1 | SM accuracy | SM F1 |
|---|---|---|---|---|
| Claude Opus 5.5 zero-shot | 0.638 | 0.571 | **0.736** | **0.720** |
| Qwen2.5-7B zero-shot | 0.483 | 0.651 | 0.566 | 0.716 |
| Qwen2.5-32B zero-shot | 0.603 | 0.676 | 0.585 | 0.633 |
| Fine-tuned DeBERTa-NLI (mean over seeds) | 0.576 | 0.461 | 0.592 | 0.601 |
| XGBoost NLI+lexical (mean over seeds) | 0.600 | 0.452 | 0.649 | 0.683 |

(n = 58 motte-and-bailey, 53 strawman.) On original examples, a strong zero-shot LLM (Claude Opus 5.5) is the
best system; the trained models do worse on originals than on the full data (see 8.6-8.7).

## 9.3 Two-stage pipeline: extract the attributed or denied claim, then verify with NLI

| Extractor | Task | Rule | Flagged (positives / negatives) | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | MB | false denial | 0.93 / 0.95 | 0.623 | 0.434 | 0.572 |
| Qwen2.5-7B | MB | narrowing | 0.93 / 0.95 | 0.567 | 0.326 | 0.485 |
| Qwen2.5-7B | SM | attributed claim not entailed | 0.60 / 0.76 | 0.461 | 0.246 | 0.436 |
| Qwen2.5-32B | MB | false denial | 0.76 / 0.74 | 0.632 | 0.381 | 0.577 |
| Qwen2.5-32B | MB | narrowing | 0.76 / 0.74 | 0.564 | 0.274 | 0.500 |
| Qwen2.5-32B | SM | attributed claim not entailed | 0.84 / 0.91 | 0.633 | 0.632 | **0.648** |

"Flagged" = share of rows where stage 1 found an attributed (strawman) or denied (motte-and-bailey) claim,
among positive and among negative rows.

Reading (Figure: `fig6_two_stage`):
- **Strawman: the idea works, but only with a strong extractor.** With Qwen-32B, checking whether A entails
  the attributed claim gives AUC 0.648 with no training at all; with Qwen-7B it is below chance (0.436).
  Extraction quality decides whether the theory-driven rule works.
- **Motte-and-bailey: it does not work.** False denial is barely above chance (0.57-0.58); narrowing is at
  chance (0.49-0.50).
- **The presence of an attribution or denial does not separate the classes:** stage 1 flags almost every
  pair, positive or negative (for example 0.93 vs 0.95). This mirrors the selection bias - every candidate
  was chosen because B uses an attribution or denial phrase - so only the entailment check carries signal.
- Thresholds were fixed in advance (0.5), not tuned; the pipeline is untrained.
