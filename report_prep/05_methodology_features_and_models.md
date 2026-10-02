# 5. Methodology: Features and Models

## Overview of the two-stage pipeline

Per the original proposal: Stage 1 is claim extraction (done manually in this
project — see file 03); Stage 2 is claim-pair relation classification, which
is what this file covers. Two feature families are computed per claim pair,
feeding a classifier; a zero-shot LLM baseline is run for comparison without
any of these features.

## Feature family 1: NLI features

Script: `src/features/nli_features.py`

**Model**: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
(Laurer et al., 2024) — a publicly released NLI model jointly trained on
MNLI, FEVER-NLI, ANLI, LingNLI, and WANLI. **Used entirely frozen, as a
feature extractor only. It is never fine-tuned on this project's data.** This
matches the proposal's explicit design choice to reuse an off-the-shelf
entailment model rather than train one from scratch, given the data budget.

**Computation**: for each claim pair, the model scores both directions:
- `entail_AB`, `neutral_AB`, `contra_AB` = P(entailment/neutral/contradiction)
  treating claim_A as premise, claim_B as hypothesis
- `entail_BA`, `neutral_BA`, `contra_BA` = the reverse direction

Six features total per pair.

**Rationale** (from the proposal): a motte-and-bailey retreat is
characterized by claim_B being a strictly weaker, narrower statement that
does *not* entail the original claim_A, while being treated as though it
does — so low `entail_BA` combined with the speaker's behavior (acting as if
defended) is the target signal. A strawman is characterized by low entailment
between the claim actually made and the claim being rebutted.

## Feature family 2: Specificity features

Script: `src/features/specificity_features.py`

**These are simple lexicon-based heuristics, not a discourse parser or any
kind of learned model.** Computed features:
- `hedge_count_A`, `hedge_count_B`, `hedge_delta` (B − A): counts of hedging
  words/phrases ("some", "often", "may", "might", "possibly", "generally",
  "i think", "not necessarily", etc. — a fixed list of ~25 terms)
- `hedge_rate_delta`: the above, normalized by claim word count
- `absolute_count_A`, `absolute_count_B`, `absolute_delta`: counts of
  absolute/universal words ("all", "every", "always", "never", "completely",
  "only", etc. — a fixed list of ~18 terms)
- `length_ratio_B_over_A`: word-count ratio
- `explicit_denial_B`: a binary indicator — does claim_B contain a phrase
  like "I never said", "I didn't say", "I'm not saying", "that's not what I
  said" (checked via regex)

## Why two separate binary tasks instead of one 3-way classifier

This is a specific methodological decision made during the project (not part
of the original plan) and needs its own clearly-reasoned paragraph in the
report. The mechanism, precisely:

- Candidate generation (file 02) defines each shape structurally: every
  motte-bailey-shaped row is, by construction, same-speaker-with-an-
  intervening-challenge; every strawman-shaped row is, by construction,
  different-speaker-direct-reply.
- This means "same speaker?" and "was there a challenge?" — the discourse
  features the original proposal planned to use — are **constants within
  each shape** (always true for motte-bailey rows, always false for strawman
  rows). They carry zero information about whether a fallacy occurred; they
  only encode which shape a row came from.
- A 3-way classifier (MOTTE_BAILEY / STRAWMAN / NONE) given these features,
  or even implicitly learning the shape from other structural cues, could
  trivially separate motte-bailey rows from strawman rows without reading any
  claim content — an artifact of how the data was constructed, not a
  measure of fallacy-detection ability.
- **Resolution**: evaluate two independent binary tasks — `MOTTE_BAILEY vs.
  NONE` (on motte-bailey-shaped rows only) and `STRAWMAN vs. NONE` (on
  strawman-shaped rows only). Discourse features are excluded from the
  feature set entirely for this reason (not because they're unhelpful in
  principle, only because they're uninformative in this specific per-shape
  framing).
- **Do not describe the final system as a 3-way classifier anywhere in the
  report** — it is two separate binary classifiers, trained and evaluated
  independently.

## Classifier: XGBoost

**Not a neural model.** Gradient-boosted decision trees, via the `xgboost`
Python package's scikit-learn-compatible API (`XGBClassifier`).

**Exact hyperparameters used** (from `src/models/train_xgboost.py` and
`src/models/ablation.py` — do not round or alter): `n_estimators=50,
max_depth=3, learning_rate=0.1`, fixed `random_state=42`, `eval_metric=
"logloss"`.

**Evaluation protocol**: 5-fold stratified cross-validation
(`StratifiedKFold`, `n_splits=5, shuffle=True, random_state=42`), per shape,
rather than a single train/test split. **Justification to state explicitly**:
with 44-58 examples per shape, a single train/test split would produce a
noisy, unreliable estimate; cross-validation averages the result over
several different splits of the same small dataset.

**Metrics reported**: accuracy, and precision/recall/F1 for the positive
class (`MOTTE_BAILEY` or `STRAWMAN` respectively), aggregated across folds
(predictions pooled across all 5 held-out folds, then scored once — this is
what the scripts actually compute, not per-fold-averaged metrics; state this
precisely since the two approaches can differ).

## Zero-shot LLM baseline

**Setup**: given only `claim_A` / `challenge` / `claim_B` text (no
structural/discourse features, no training on this dataset), predict the
per-shape binary label. Run on the 91 real examples only (the 20 synthetic
examples were not included in this baseline's evaluation set, since their
labels were visible in the same context the predictions were made in — see
limitation below).

**Blinding procedure used**: predictions were generated from a label-stripped
copy of the data, with the true label never consulted until after a
prediction was already recorded — the standard procedure for a baseline to
be meaningful.

**Required limitation — state this explicitly, do not omit or bury it**: the
same model that performed the manual claim extraction for this dataset
(file 03) also produced these zero-shot predictions, within the same
overall project context. This is **not a fully independent, arms-length
zero-shot evaluation** — residual familiarity with specific examples from the
extraction process cannot be ruled out. The reported 0.750/0.800 F1 numbers
should be read with this caveat attached, not presented as a clean,
independent LLM-capability measurement. A fully independent baseline would
require a separate model/session with no prior exposure to these examples —
named explicitly as a gap, not silently left out.

## Code and file references

- `src/features/nli_features.py` — NLI feature extraction
- `src/features/specificity_features.py` — specificity feature extraction
- `src/models/train_xgboost.py` — main XGBoost training/eval script (takes a dataset name argument, e.g. `dataset_consolidated` for 91-real or `dataset_final` for 111-combined)
- `src/models/ablation.py` — feature-ablation variant (NLI-only / specificity-only / combined), also saves out-of-fold predictions used in error analysis (file 08)
- Zero-shot predictions: `data/annotations/zeroshot_baseline_predictions.csv`
