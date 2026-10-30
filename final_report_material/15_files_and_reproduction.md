# 15. Files and reproduction

## Data files
| File | Contents |
|---|---|
| `data/annotations/annotated_batch_{1,2,3,4,5,6}_to_label.csv` | Hand-labeled batches. Note: `annotated_batch_5_to_label.csv` contains the rows of sampling batch 7 (pair ids m151-s175); sampling batches 5 and 8 were not labeled. |
| `data/annotations/dataset_consolidated_v2.csv` | 188 real labeled pairs |
| `data/annotations/dataset_consolidated_v2_label_audit.csv` | Fate of every annotated row (kept / dropped and why) |
| `data/annotations/synthetic_examples.csv` | 20 hand-written pairs |
| `data/annotations/dataset_final_v2.csv` | 188 real + 20 hand-written |
| `data/augmented/*_variants_accepted.csv` | 100 + 375 generated variants with `source_pair_id` and `kind` |
| `data/annotations/dataset_augmented_all_v2.csv` | **Final dataset, 683 pairs** |
| `data/annotations/dataset_augmented_all_v2_specificity_features.csv` | Lexical features |
| `finetune/session_results/features/dataset_augmented_all_v2_nli_features.csv` | NLI features |

## Results files
| File | Contents |
|---|---|
| `finetune/session_results/<run>/predictions.csv` | Held-out predictions per seed and fold (prob, pred, masked prob, masked pred) |
| `finetune/session_results/xgb_grouped/predictions.csv` | XGBoost predictions on the same folds |
| `finetune/session_results/probing/probe_results.csv` | Probe metrics per encoder, task and layer |
| `llm_bundle/llm_results/llm/<model>/` | Zero-shot predictions, extractions, NLI stage 2, prompts, run logs |
| `results_final/RESULTS.md` and `table*.csv` | Every table in this folder |

## Code
| Step | Script |
|---|---|
| Candidates and filter | `src/extraction/build_candidates.py`, `filter_candidates.py` |
| Sampling and merging | `src/annotation/prepare_batch.py`, `merge_for_labeling.py` |
| Consolidation | `src/annotation/consolidate_v2.py`, `build_dataset_final_v2.py` |
| Variants | `src/augmentation/generate_variants.py`, `collect_accepted.py` |
| Features | `src/features/specificity_features.py`, `finetune/features_gpu.py` |
| XGBoost | `src/models/xgb_grouped.py` |
| Fine-tuning, masking | `finetune/train_cv.py`, `finetune/triggers.py` |
| Probing | `finetune/probe_layers.py` |
| GPU session driver | `finetune/run_session.py` |
| LLM baselines and two-stage | `llm_bundle/llm_run.py`, `nli_stage2.py`, `llm_session.py` |
| Tables and figures | `src/analysis/final_results.py`, `llm_results.py`, `orig_only.py`, `paper_figures.py` |

## Rerunning the analysis (local)
```bash
python src/models/xgb_grouped.py
cd src/analysis
python final_results.py && python llm_results.py && python orig_only.py && python paper_figures.py
```
Full pipeline and GPU commands: `README.md` at the repository root.

## Repository
https://github.com/Ad-S/relational-fallacy-detection (branch `master`).
