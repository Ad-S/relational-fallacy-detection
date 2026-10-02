# Report Preparation Folder — Index

This folder documents everything done on this project, step by step, in
enough detail to write the mid-submission report from. Every number in every
file here has been directly verified against the actual files in this repo
(not recalled from memory) as of 2026-10-02.

**How to use this folder**: read the files in order. Each one covers one
phase of the project. Pull the facts/numbers/examples you need into your own
sentences — these files are reference material, not the report itself.

## Files in this folder

| File | Covers |
|---|---|
| `01_project_motivation_and_formulation.md` | What the project is, why it matters, formal task definition, research questions |
| `02_data_pipeline_and_corpus.md` | CMV/ConvoKit corpus, candidate generation code and logic, raw numbers at each filtering stage |
| `03_claim_extraction_process.md` | How raw Reddit text became clean claim triples, batch-by-batch outcomes |
| `04_annotation_protocol_and_dataset.md` | Labeling rules, guideline examples, synthetic data, final dataset composition |
| `05_methodology_features_and_models.md` | NLI features, specificity features, per-shape binary framing rationale, XGBoost setup, zero-shot baseline setup |
| `06_experimental_results.md` | All result tables, exact numbers, confusion matrices |
| `07_ablation_and_deeper_analysis.md` | Feature ablation, NLI visualization (null result), real-vs-synthetic distribution analysis |
| `08_error_analysis.md` | Error categories, shared lexical-shortcut finding, representative examples |
| `09_literature_review_notes.md` | Paper-by-paper facts for every cited work |
| `10_limitations_and_future_work.md` | Honest limitations list, dated remaining-work timeline to Oct 31 |
| `11_figures_index.md` | Every figure, what it shows, what it's honest about, suggested placement |
| `12_repository_map.md` | Every script/file in the repo and what it does, so you can cite/link to exact code |

## What's already written for you elsewhere in the repo (don't duplicate)

- `data/annotations/ANNOTATION_GUIDELINE.md` — the actual annotation guideline, with real worked examples
- `data/annotations/ERROR_ANALYSIS.md` — the full error analysis writeup with categories and quotes
- `../REPORT_CONTENT_DOSSIER.md` — an earlier, section-by-section version of this same material, organized to match a 15-section ACL paper structure directly (if you prefer working from one structure-mapped file instead of these topic files, use that one instead — it has the same verified numbers)

## Quick fact sheet (the numbers you'll use constantly)

- Corpus: 293,297 utterances, 3,051 conversations (ConvoKit, Tan et al. 2016 CMV corpus)
- Raw structural candidates: 376,714 (119,452 motte-bailey-shaped + 257,262 strawman-shaped)
- Keyword-filtered candidates: 5,078 (3,109 motte-bailey + 1,969 strawman)
- Sampled for extraction: 300, across 6 batches
- Usable after extraction: 240
- Hand-labeled real: 91 (20 MOTTE_BAILEY, 23 STRAWMAN, 48 NONE)
- Synthetic added: 20 (8 MOTTE_BAILEY, 6 STRAWMAN, 6 NONE)
- **Final dataset: 111** (28 MOTTE_BAILEY, 29 STRAWMAN, 54 NONE)
- XGBoost F1 (91 real): Motte 0.579, Strawman 0.591
- XGBoost F1 (111 combined): Motte 0.590, Strawman 0.586
- Zero-shot LLM F1 (91 real): Motte 0.750, Strawman 0.800
- Ablation: NLI-only accuracy 0.724 (Motte) vs. NLI+Specificity 0.569 (Motte)
- `explicit_denial_B` rate: 37.4% real vs. 0.0% synthetic
- Repo: https://github.com/Ad-S/relational-fallacy-detection (public)
