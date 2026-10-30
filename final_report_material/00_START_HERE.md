# Final report material - start here

Everything done in the project, the exact numbers, and all figures and tables, in one place.
Read the files in order; each covers one part of the work. All numbers were computed from the files in
this repository (`results_final/`, `finetune/session_results/`, `llm_bundle/llm_results/`, `data/`).

| File | Covers |
|---|---|
| `01_project_overview_and_research_questions.md` | Task, formulation, the central question, research questions, contributions |
| `02_related_work_and_gap.md` | Prior work grouped by theme, the precise research gap, what is and is not novel |
| `03_data_pipeline_and_annotation.md` | Corpus, candidate generation, trigger filter, sampling, claim extraction, annotation, consolidation |
| `04_dataset_expansion.md` | Hand-written examples, generated variants, final 683-pair dataset, grouping |
| `05_features_models_methods.md` | NLI and lexical features, XGBoost, fine-tuning, probing, trigger masking, LLM baselines, two-stage pipeline |
| `06_evaluation_protocol.md` | Grouped cross-validation, seeds, metrics, baselines, confidence intervals |
| `07_results_main.md` | Main comparison and paired tests |
| `08_analysis_what_models_use.md` | B-only ablation, trigger masking, probing, learning curve, learning-rate sensitivity, data source, flip pairs, originals-only training |
| `09_llm_baselines_and_two_stage.md` | Zero-shot LLMs and the extract-then-verify pipeline |
| `10_qualitative_examples.md` | Real examples with model predictions, for the error analysis |
| `11_limitations_and_future_work.md` | Limitations and future directions (for the section before the references) |
| `12_compute.md` | GPU runs, hardware, timings, costs (for the required compute-credit section) |
| `13_changes_since_mid_submission.md` | What changed since the mid-term report, including corrections |
| `14_figures_guide.md` | Every figure: what it shows, the takeaway, where to place it |
| `15_files_and_reproduction.md` | Where everything lives and how to rerun it |
| `figures/` | All paper figures (PDF for LaTeX, PNG for slides) |
| `tables/` | `RESULTS.md` (every result table) and the CSV behind each table |

## Fact sheet (the numbers used most often)

**Data**
- Corpus: ConvoKit Winning Arguments (r/ChangeMyView): 293,297 utterances, 3,051 conversations, 34,911 speakers.
- Structural candidates: 119,452 motte-and-bailey-shaped, 257,262 strawman-shaped (376,714 total).
- After trigger-phrase filter: 3,109 + 1,969 = 5,078.
- Sampled 8 batches x 50 = 400 candidates; 6 batches (300) annotated.
- Final dataset: **683 claim pairs** = 188 real (hand-labeled) + 20 hand-written + 475 generated variants.
- Motte-and-bailey task: 353 pairs, 127 positive (36.0%); majority accuracy 0.640.
- Strawman task: 330 pairs, 180 positive (54.5%); majority accuracy 0.545.

**Main results** (grouped 5-fold CV, mean over seeds; full table in `07_results_main.md`)

| Model | MB accuracy | MB F1 | MB AUC | SM accuracy | SM F1 | SM AUC |
|---|---|---|---|---|---|---|
| Majority class | 0.640 | 0.000 | 0.500 | 0.545 | 0.706 | 0.500 |
| XGBoost (NLI + lexical) | 0.638 | 0.412 | 0.592 | 0.682 | 0.725 | 0.701 |
| DeBERTa plain, fine-tuned | 0.646 | 0.531 | 0.629 | 0.652 | 0.692 | 0.699 |
| **DeBERTa-NLI, fine-tuned** | **0.665** | **0.538** | **0.674** | **0.701** | **0.736** | **0.770** |
| DeBERTa-NLI, B only | 0.587 | 0.514 | 0.616 | 0.655 | 0.699 | 0.704 |
| Qwen2.5-32B zero-shot | 0.555 | 0.596 | - | 0.706 | 0.757 | - |
| Qwen2.5-7B zero-shot | 0.360 | 0.529 | - | 0.555 | 0.710 | - |

(MB = motte-and-bailey, SM = strawman. Zero-shot LLMs give labels only, so no AUC.)

**Headline findings**
1. Fine-tuned DeBERTa-NLI ranks examples best on both tasks (AUC 0.674 / 0.770); accuracy gains over XGBoost are small and not significant.
2. The full exchange matters for motte-and-bailey: A, C, B beats B alone by +0.063 accuracy (95% CI +0.007 to +0.116).
3. Trigger-phrase reliance for motte-and-bailey only: removing the phrase drops accuracy on affected rows 0.645 -> 0.581; strawman unchanged (0.671 -> 0.670).
4. NLI pretraining helps strawman (AUC 0.770 vs 0.699), not motte-and-bailey.
5. Frozen-encoder probing: the signal sits in middle and late layers; for motte-and-bailey the best frozen layer (AUC 0.718) matches the fine-tuned model.
6. Learning curves still rise at 100% of the training data.
7. On the 208 original examples, motte-and-bailey is close to chance for all trained models; generated variants clearly help strawman (AUC 0.592 -> 0.694).
8. Qwen2.5-32B zero-shot matches fine-tuning on strawman; Qwen2.5-7B labels everything a fallacy.
9. The theory-driven two-stage pipeline works only for strawman and only with the stronger extractor (AUC 0.648 with 32B; 0.436 with 7B).
