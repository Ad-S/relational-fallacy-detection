# 14. Figures guide

All in `figures/` as PDF (use in LaTeX) and PNG (use in slides). Colours are consistent across figures:
dark blue = fine-tuned DeBERTa-NLI, light blue = plain DeBERTa, green = XGBoost, orange = Qwen2.5-32B,
light orange = Qwen2.5-7B, grey = B-only / reference. Dotted line = chance (AUC 0.5).

| File | Shows | Takeaway | Suggested place |
|---|---|---|---|
| `fig0_dataset` | 683 pairs by source (real, hand-written, generated) and label | Composition of the data; generated variants are the majority | Data section |
| `fig1_roc` | ROC curves of XGBoost, plain DeBERTa and DeBERTa-NLI per task; Qwen zero-shot as points | DeBERTa-NLI ranks best on both tasks; strawman is easier; Qwen-7B sits at "always fallacy"; Qwen-32B over-predicts motte-and-bailey | Main results |
| `fig2_what_the_model_uses` | Left: AUC with B only vs the full exchange. Right: accuracy on trigger-bearing rows before/after removing the phrase | Full exchange helps both tasks; trigger removal hurts motte-and-bailey but not strawman | Main analysis |
| `fig3_probing` | Probe AUC per layer of frozen DeBERTa-NLI and plain DeBERTa; dashed = fine-tuned | Signal in middle/late layers; NLI encoder stronger for strawman; frozen layer 18 matches fine-tuning on motte-and-bailey | Analysis |
| `fig4_learning_curve` | AUC vs share of training data; dashed = XGBoost with all data | Still rising at 100%; fine-tuning overtakes full-data XGBoost at 50-75% of the data (motte-and-bailey) and 25-50% (strawman) | Analysis |
| `fig5_by_source` | Accuracy on real, hand-written and generated pairs for XGBoost, DeBERTa-NLI and Qwen-32B | On real pairs fine-tuning is not better than XGBoost; the advantage comes from generated pairs | Analysis / limitations |
| `fig6_two_stage` | AUC of the extract-then-verify pipeline with a 7B vs 32B extractor | Works for strawman only with the stronger extractor; fails for motte-and-bailey | LLM section |
| `fig7_variants_help` | AUC on the 208 originals when training on originals only vs with variants | Variants help strawman on real data (0.59 -> 0.69); motte-and-bailey stays near chance | Analysis |

If space is short in 8 pages: keep fig1, fig2, fig3 and fig7 in the main text; put fig0, fig4, fig5 and fig6 in
the appendix (the appendix may be ignored by the evaluator, so the key evidence should stay in the main text).

Mid-term figures and slides material: `../presentation_figures/figures/`.
