# 12. Compute (for the required compute-credit section)

Course sponsorship: Jarvislabs.ai credits, INR 5,880 per team. All GPU work ran on Jarvislabs on-demand
containers (pytorch template, region IN2), launched with the `jl` CLI; each job ran unattended from a
self-contained directory (`finetune/`, `llm_bundle/`).

## Runs

| Run | GPU (rate) | What it did | Duration | Cost |
|---|---|---|---|---|
| Smoke test | A100 40GB (INR 84.24/h) | Environment check, 3 training steps per task | ~23 s of jobs + setup | INR 1.38 (run) |
| Failed launch | A100 80GB (INR 140.94/h) | Instance unreachable (no SSH key on the account yet); destroyed | a few minutes | ~INR 14-20 (check dashboard) |
| Main session | H200 (INR 434.97/h) | NLI features for 683 pairs; fine-tuning grid (DeBERTa-NLI x5 seeds, plain, B-only, 2 learning rates, 3 learning-curve points); probing of 2 frozen encoders; RoBERTa attempt | jobs 13:56-15:13 UTC (77 min) + setup; then idle until destroyed | see dashboard |
| LLM session | H200 | Qwen2.5-7B and -32B zero-shot + claim extraction on 683 pairs; NLI stage 2 | ~11 min of jobs (incl. 32B download) + package install | INR 87.32 (run) + resume/download |
| Originals-only | H200 | DeBERTa-NLI trained/evaluated on the 208 originals, 3 seeds | 3.5 min of jobs | INR 29.00 (run) |

**Total spent = 5,880 - remaining grant** (`jl status`, "Grants"); take the per-instance costs from the
Jarvislabs dashboard for the final table.

## Job timings inside the main session (from `finetune/session_results/session_summary.json`)
| Step | Minutes |
|---|---|
| Smoke tests (4 steps) | 1.2 |
| NLI features (683 pairs, both directions) | 0.2 |
| DeBERTa-NLI fine-tuning, 5 seeds x 5 folds x 2 tasks | 16.9 |
| Probing (2 encoders x 25 layers x 2 tasks x 3 seeds) | 0.8 |
| B-only fine-tuning, 3 seeds | 10.5 |
| Plain DeBERTa fine-tuning, 3 seeds | 9.9 |
| RoBERTa-MNLI (failed, NaN) | 0.4 |
| Learning rate 1e-5 / 3e-5, 3 seeds each | 9.9 / 10.0 |
| Learning curve 25% / 50% / 75% | 3.2 / 5.8 / 8.0 |

Per-step speed: about 0.09 s per training step, peak GPU memory about 10.6 GB (DeBERTa-v3-large, batch 8).
LLM session: model loading 32 s (7B) and 329 s (32B, including download); generation for all 683 pairs and both
prompts 7.7 s (7B) and 34.3 s (32B) with vLLM.

## Points for the justification
- Every run maps to a result in the report: fine-tuning comparisons, the learning-curve and learning-rate
  analyses, probing, the open-weight zero-shot baselines and the two-stage pipeline.
- Unattended sessions with a hard time limit and automatic pausing kept idle time low; a smoke pass before
  each session caught failures cheaply (it isolated the RoBERTa failure to one 0.4-minute step).
- The H200 was used because A100 capacity was unavailable at launch time; the 32B model needs more than
  40 GB of GPU memory.
- CPU-only work (data pipeline, lexical features, XGBoost, analysis, figures) ran locally.
