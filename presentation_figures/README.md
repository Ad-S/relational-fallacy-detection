# Presentation figures (mid-term evaluation)

Every number plotted is computed from CSVs in `data/annotations/`. Nothing is hard-coded.
The only transcribed numbers are in `reported_mid_report_tables.csv` (mid-report Tables 3 and 4);
scripts use them to **assert** that the recomputed values match the report, never as plot input.

Regenerate everything: `bash presentation_figures/make_all.sh` (uses `venv/bin/python`, run from anywhere).

## What is original vs. new

| Kind | Where |
|---|---|
| **Reported in the mid-report** (recomputed, exact match) | Table 3/4 numbers in figs 1, 5; the real + synthetic panel of fig 2 (Table 6); the 48/9/29/10 error counts (listed in `derived/error_overlap_counts.csv`) |
| **New for the presentation, from the existing experiment code** | Real-only ablation + real-only XGBoost OOF predictions (`src/models/ablation.py dataset_consolidated`, unmodified) |
| **Audit-derived statistics** (not in the report) | Bootstrap CIs and McNemar test (fig 1), Wilson CIs (fig 3), the real-only error overlap 45/8/26/11 (fig 4) |

## Figures

| File | Script | Reads | Shows |
|---|---|---|---|
| `figures/01_model_comparison.png` | `01_model_comparison.py` | `zeroshot_baseline_predictions.csv`, `dataset_consolidated_xgboost_oof_predictions.csv`, `dataset_consolidated.csv` | Accuracy and F1 for XGBoost vs zero-shot LLM on the 91 real examples, per shape, with majority-class rate, 95% bootstrap CIs (10,000 paired resamples, seed 0) and the paired accuracy gap with McNemar exact p in the footnote. |
| `figures/02_ablation_accuracy.png`, `02_ablation_f1.png` | `02_ablation_two_panel.py` | `dataset_consolidated_ablation_results.csv` (new), `dataset_final_ablation_results.csv`, `dataset_consolidated.csv`, `dataset_final.csv` | NLI-only / specificity-only / full-model XGBoost, real-only (left) vs real + synthetic (right), with majority rates. No intervals (single CV run). |
| `figures/03_explicit_denial_rates.png` | `03_explicit_denial_rates.py` | `dataset_final.csv`, `dataset_final_specificity_features.csv` | **Exploratory.** Share of examples where B matches the `explicit_denial_B` regex, by source x shape x label, with Wilson 95% CIs. Descriptive only. |
| `figures/04_error_overlap.png` | `04_error_overlap.py` | same as fig 1 + `dataset_final_xgboost_oof_predictions.csv` (only to report the sec 8.4 run's counts) | Stacked counts of both wrong / XGBoost only / LLM only / both correct on the 91 real examples, using the real-only XGBoost run. |
| `figures/05_confusion_matrices.png` (backup) | `05_confusion_matrices.py` | same as fig 1 | The four confusion matrices of Table 4 (real-only XGBoost, LLM). |
| `figures/06_llm_vs_nli_real.png` | `06_llm_vs_nli_real.py` | same as fig 1 + `dataset_consolidated_{nli,specificity}_features.csv`, `dataset_consolidated_ablation_results.csv` | Zero-shot LLM vs NLI-only XGBoost vs full XGBoost on the 91 real examples, with paired bootstrap CIs and McNemar (LLM vs NLI-only) in the footnote. NLI-only predictions are recomputed with `ablation.run_one` and asserted equal to the saved ablation CSV. **Real-only by necessity**: the LLM was never run on the 20 synthetic examples. |
| `figures/07_llm_rerun_comparison_{accuracy,f1}.png` | `07_llm_rerun_comparison.py` | `llm_rerun/zeroshot_rerun_predictions.csv` + XGBoost OOF files for both datasets + ablation CSVs | **NEW scripted LLM run** (Claude Opus 5.5) vs NLI-only and full XGBoost, real-only (n=91) and real + synthetic (n=111). Extra numbers (synthetic-only accuracy, agreement with the original LLM labels) in `derived/llm_rerun_extra.csv`. |

`derived/` holds the numbers behind each figure: `model_comparison_stats.csv`,
`model_comparison_paired_tests.csv`, `ablation_long.csv`, `explicit_denial_rates.csv`,
`error_overlap_counts.csv`, `error_overlap_per_example.csv` (pair_id, gold, predictions, overlap category;
useful for picking qualitative examples).

## New zero-shot LLM run (`llm_rerun/`)

`llm_rerun/run_zeroshot_llm.py` runs a **new** zero-shot baseline on all 111 pairs. It is not the mid-report's baseline.
- Model: `claude-opus-5-5` through headless Claude Code (`claude -p`), the user's own login. One fresh session per example,
  replaced system prompt, no tools, no session persistence, no MCP servers, launched from an empty temp directory (no CLAUDE.md / project memory).
- Prompts: `llm_rerun/prompts.py`. Definitions paraphrase the annotation guideline's rules; its worked examples are excluded because they come from the dataset.
- Inputs: `pair_id, type_candidate, source, claim_A, challenge, claim_B` only; labels are never read by the runner.
- Outputs: raw CLI JSON per example (`llm_rerun/raw/`, includes prompt, model usage, cost), `llm_rerun/zeroshot_rerun_predictions.csv` (label + one-sentence reason), `llm_rerun/run_log.txt`.
- Limits: single run; no temperature control through the CLI; still a Claude model (a different model from the claim extractor, Claude Sonnet 5, but the same family).
- Re-running the script skips examples that already have a valid raw file; `make_all.sh` does **not** call the LLM.

## Things to state when presenting

1. **Error overlap differs from the report's section 8.4.** The report's 48 / 9 / 29 / 10 (any wrong / both / XGBoost only / LLM only)
   came from XGBoost predictions of the *real + synthetic* run, restricted to the 91 real rows. Fig 4 uses the real-only run
   that Table 3 reports: 45 / 8 / 26 / 11. Both are in `derived/error_overlap_counts.csv`. Examples such as s051 flip between runs;
   check `derived/error_overlap_per_example.csv` before citing one as "both systems miss".
2. **Ablation.** The report's headline NLI-only advantage (motte-and-bailey accuracy 0.72 vs 0.57) is from the real + synthetic data.
   On real data alone it is 0.66 vs 0.64.
3. **The LLM baseline is not fully reproducible.** There is no script, prompt or raw output in the repo; only final labels
   (`zeroshot_baseline_predictions.csv`) were saved, and the same model extracted the claims. Nothing here regenerates those predictions.
4. **Uncertainty.** n = 44 / 47: the accuracy CIs of XGBoost include the majority-class rate in both shapes, and the LLM-vs-XGBoost accuracy gap
   is not significant at 0.05 (McNemar exact p = 0.21 motte-and-bailey, 0.078 strawman).
5. **Fig 3 is exploratory.** The regex matches 'I never said', 'I'm not saying', 'that's not what I said' etc. but not strawman triggers, so it is 0 on
   every strawman-shaped row by construction. In motte-and-bailey it is high for both positives (16/20) and NONE (18/24). It does not show that any
   model relies on it; the trigger-masking ablation has not been run.
6. Claims A/B/C are the LLM-extracted condensations, not verbatim Reddit text.

## Requirements

`venv/bin/python` with pandas, numpy, matplotlib, scikit-learn, xgboost (all already in the repo venv).
