# 4. Dataset expansion: hand-written and generated examples, final dataset

## 4.1 Hand-written examples (`data/annotations/synthetic_examples.csv`)
- 20 pairs written by hand following the same shapes and labeling rules.
- Motte-and-bailey-shaped: 8 MOTTE_BAILEY, 6 NONE. Strawman-shaped: 6 STRAWMAN, 0 NONE.
- None of them contains a selection trigger phrase in B.
- Each gets its own group id (`handwritten_<pair_id>`) so cross-validation spreads them across folds.

## 4.2 Generated variants (`src/augmentation/`)
Generated with Claude Sonnet 5.5; one isolated session per seed example (fixed system prompt, no tools);
prompts in `src/augmentation/prompts.py`, raw outputs in `data/augmented/raw*/`. Definitions given to the
generator follow the annotation guideline; no dataset examples are included in the prompt.

**From each hand-written example (5 variants):**
- 2 x **paraphrase** - the same exchange reworded, same meaning, same label.
- 2 x **topic transfer** - the same logical move on a different everyday topic, same label.
- 1 x **flip** - A (and C) copied word for word, B minimally rewritten so the label flips
  (MOTTE_BAILEY <-> NONE, STRAWMAN -> NONE).

**From each real example (2 variants):**
- 1 x paraphrase and 1 x topic transfer, both keeping the human label (no flips).

Automatic checks on every variant: correct number and kinds, label consistent with the kind, non-empty
claims, challenge present for motte-and-bailey and absent for strawman, flips keep A and C unchanged.
All variants passed. One real seed (m096, which rationalises attraction to minors) produced only its
topic-transfer variant; the generator declined to paraphrase it.

| Variant set | Paraphrase | Topic transfer | Flip | Total |
|---|---|---|---|---|
| From 20 hand-written | 40 | 40 | 20 | 100 |
| From 188 real | 187 | 188 | 0 | 375 |
| **Total** | 227 | 228 | 20 | **475** |

Variant labels were not individually re-checked by hand (`label_verified=False` in the data). Real-seed
variants inherit the human label of their source; flip labels are generated.

## 4.3 Final dataset (`data/annotations/dataset_augmented_all_v2.csv`, 683 pairs)

| Source | MB-shaped: MOTTE_BAILEY | MB-shaped: NONE | SM-shaped: STRAWMAN | SM-shaped: NONE | Total |
|---|---|---|---|---|---|
| Real (CMV) | 27 | 63 | 50 | 48 | 188 |
| Hand-written | 8 | 6 | 6 | 0 | 20 |
| Variants of hand-written | 38 | 32 | 24 | 6 | 100 |
| Variants of real | 54 | 125 | 100 | 96 | 375 |
| **Total** | **127** | **226** | **180** | **150** | **683** |

- Motte-and-bailey task: 353 pairs, 36.0% positive (majority accuracy 0.640).
- Strawman task: 330 pairs, 54.5% positive (majority accuracy 0.545).
- The flips supply the only strawman-shaped NONE examples among the hand-written material.

## 4.4 Grouping (essential for honest evaluation)
- Every variant carries its source example's `conversation_id`.
- Cross-validation is grouped by `conversation_id`, so an example and all its variants (and all pairs from
  the same thread) always fall in the same fold - a model is never trained on a paraphrase and tested on
  its original.
- Groups: 97 for motte-and-bailey, 101 for strawman. Confidence intervals resample these groups.

## 4.5 Mid-term vs final data
| | Mid-term | Final |
|---|---|---|
| Real labeled pairs | 91 (batches 1-3) | 188 (batches 1-4, 6, 7) |
| Hand-written | 20 | 20 |
| Generated variants | 0 | 475 |
| Total | 111 | 683 |
| Cross-validation | stratified 5-fold, one seed | grouped 5-fold, 3-5 seeds |
