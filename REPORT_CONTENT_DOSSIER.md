# Content Dossier for Mid-Submission Report

Every number below is verified directly against the repo's result files as of
2026-10-02 (commands used to verify are in the project's conversation history;
all CSVs are in `data/annotations/`). This is organized by your requested
15-section structure. It is facts, numbers, structure, and literature notes —
**not finished prose** — written that way on purpose. Turning each bullet
list into connected sentences is the part that has to be yours.

---

## 1. Abstract — facts to compress into ~150 words

- Problem: existing fallacy detection (MAFALDA, Goffredo et al.) is span-level;
  motte-and-bailey and strawman are relational and invisible to span classifiers.
- Contribution: formalize both as cross-turn claim-pair classification; build
  and release a 111-example seed dataset (91 real CMV + 20 synthetic) via a
  reply-structure + keyword-filtered candidate pipeline from ConvoKit's CMV
  corpus (293,297 utterances); evaluate a frozen-NLI + lexical-feature XGBoost
  classifier against a zero-shot LLM baseline, per fallacy type.
- Headline results: zero-shot LLM outperforms XGBoost on both shapes (Motte F1
  0.750 vs 0.579; Strawman F1 0.800 vs 0.591, on 91 real examples); ablation
  shows NLI-only features outperform the combined feature set (0.724 vs 0.569
  accuracy, Motte); error analysis finds a shared lexical-shortcut
  vulnerability in both models.
- Framing: this is a mid-submission — dataset + pipeline + baselines done;
  transformer fine-tuning, trigger-phrase-masking ablation, and larger-scale
  annotation are explicitly future work (see §12/§15).

---

## 2. Introduction — facts and argument chain

- Open on the general claim: computational argument mining has built strong
  infrastructure for span-level fallacy classification (MAFALDA; Goffredo et
  al. 2022, 2023).
- State the structural limitation precisely: a motte-and-bailey retreat is not
  fallacious in either sentence alone — the first claim is just a claim, the
  retreat is just a narrower claim. The fallacy exists only in the *relation*
  (treating the second as still defending the first). Same logic for strawman:
  the distortion is a mismatch between two claims, not a property of either
  one read in isolation.
- Use the real example from `ANNOTATION_GUIDELINE.md` to ground this
  concretely (verbatim, already extracted from the dataset, conversation
  `t3_2rnfn0`):
  > claim_A: "If these measures are expected and treated as reasonable, then
  > it serves to make the crime more acceptable..."
  > challenge: "...Leaving our doors and windows unlocked is not going to
  > reduce theft, it's going to increase it."
  > claim_B: "I never said that leaving them unlocked reduces theft... You
  > said [X]. Essentially, that locking doors increases crime."

  Neither claim_A nor claim_B is independently fallacious; the retreat-while-
  still-claiming-to-defend-the-original is the fallacy.
- State the two-part gap: (1) no existing dataset treats claim pairs as the
  annotation unit; (2) this is a data-representation gap, not something a
  bigger span classifier fixes.
- State contributions plainly, hedged correctly: "to our knowledge" language
  per your original proposal — do not claim first-ever.
- Close with a one-line map of the paper's structure.

---

## 3. Problem Formulation

**Formal statement.** Given a conversation thread, a candidate pair consists
of claim_A, claim_B, and (for motte-bailey candidates only) an intervening
challenge turn. Task: predict one of `{MOTTE_BAILEY, NONE}` for motte-bailey-
shaped candidates, or `{STRAWMAN, NONE}` for strawman-shaped candidates.

**Why two separate binary tasks, not one 3-way classifier** (this is a
required, explicit paragraph — state the mechanism precisely):
- Candidate generation defines the shape structurally: motte-bailey
  candidates are, by construction, same-speaker-with-an-intervening-challenge;
  strawman candidates are, by construction, different-speaker-direct-reply.
- Therefore "same speaker?" and "was there a challenge?" are not predictive
  features within a shape — they are *constants* within each shape (always
  true for motte-bailey rows, always false for strawman rows).
- A 3-way classifier given these as features would trivially separate
  motte-bailey from strawman from the structural metadata alone, without
  reading claim content at all — an artifact of data construction, not a
  measure of fallacy-detection ability.
- Hence: two binary fallacy-vs-NONE tasks, evaluated and reported separately.

**Research questions** (use these five, lightly rephrased if you like, but
keep the content):
- RQ1: Can semantic (NLI) and lexical (specificity) features distinguish
  genuine motte-bailey/strawman pairs from superficially similar non-fallacious
  exchanges?
- RQ2: How does this lightweight feature-based classifier compare to a
  zero-shot LLM baseline on the same task?
- RQ3: Of the two feature families, which contributes more signal?
- RQ4: Does adding hand-written synthetic examples to the training data
  improve performance?
- RQ5: What failure modes are shared across model classes, and what do they
  reveal about the task's difficulty?

---

## 4. Background

- **Toulmin (1958)**: claim/evidence/warrant decomposition — gives the
  principled notion of "claim" as the unit to extract and compare. Use to
  justify why claim extraction (not raw utterance comparison) is Stage 1.
- **Dung (1995)**: abstract argumentation frameworks, arguments as graph nodes
  with attack/support edges. Use to justify framing a fallacy as an edge-type
  between two claim-nodes rather than a node property — this is the direct
  conceptual ancestor of "claim-pair classification."
- **NLI as a task** (Bowman et al. 2015, SNLI): entailment/neutral/
  contradiction between a premise and hypothesis — briefly define it formally
  here since Section 6 uses it as a feature source, not before.
- Define motte-and-bailey and strawman as rhetorical/informal-logic concepts
  (motte = defensible fallback position; bailey = the bolder, actually-held
  position) before formalizing them computationally in §3.

---

## 5. Related Work — paper-by-paper factual backbone

Write connected prose from these; do not copy as a list of disconnected
summaries. Each entry: problem / dataset / method / result / relevance to
this project / gap this project addresses.

**Helwe et al. 2024 (MAFALDA)**
- Problem: unify >30 fallacy types into one shared taxonomy; benchmark
  LLMs/classifiers on fallacy detection and classification.
- Method/dataset: span-level annotation over curated text; evaluates multiple
  LLMs and classifiers against the taxonomy.
- Relevance: the clearest instance of the span-level paradigm this project
  departs from — a span is labeled fallacious or not, independent of any
  other span.
- Gap: cannot represent a fallacy whose fallaciousness depends on a
  relationship between two non-adjacent, individually-non-fallacious spans.

**Goffredo et al. 2022 (IJCAI)**
- Problem: fallacious argument classification in political debates.
- Dataset: annotated transcripts from 31 U.S. presidential debates, 1,628
  fallacious arguments.
- Method: Transformer-based classifiers; finds argument components/relations
  useful.
- Relevance: same domain-adjacent precedent (debate transcripts) as your
  proposal's second planned data source; shows relational/structural features
  help even in a span-classification setup.
- Gap: still predicts a fallacy label per argument/span, not per claim-pair
  relation.

**Goffredo et al. 2023 (EMNLP)**
- Problem/method: extends the above; explicitly represents claims, premises,
  support/attack relations, combines Transformer representations with
  engineered features.
- Relevance: closest prior work in spirit — explicit argumentative structure
  + engineered features + Transformer, echoed in this project's XGBoost
  (engineered features) vs. zero-shot LLM (text-only) comparison.
- Gap: the "relation" here is support/attack between argument components
  within standard argument mining, not a fallacy defined BY a cross-turn
  relation.

**Jin et al. 2024 (CoCoLoFa)**
- Problem: scale fallacy-labeled dataset construction via LLM-assisted
  crowdsourcing (news comments).
- Relevance: methodological precedent for this project's own seed dataset +
  synthetic augmentation plan (Jin et al. use LLM-generated examples from a
  human seed set; this project used hand-written synthetic examples from a
  human seed set — same structural idea, different generation mechanism; be
  explicit about this difference, don't conflate them).
- Gap: doesn't address the relational task.

**Ruggeri et al. 2025**
- Problem: fallacy judgments carry genuine annotator disagreement, not one
  ground truth.
- Relevance: direct justification for this project's `UNCERTAIN` label and
  for treating the solo-annotator limitation honestly rather than presenting
  labels as objective ground truth. Also motivates the (not yet run)
  LLM-as-second-rater agreement check in Future Work.
- Gap: addresses label variation for span-level fallacy judgments, not
  relational ones.

**Laurer et al. 2024**
- The companion paper for the frozen DeBERTa-v3-large-mnli-fever-anli-ling-
  wanli NLI model used in §6 as a feature extractor. State plainly: this model
  is used *frozen*, as a feature extractor only — it is not fine-tuned, and
  this project does not train an NLI model.

**Chang et al. 2020 (ConvoKit) / Tan et al. 2016 (Winning Arguments / CMV)**
- ConvoKit provides the programmatic access layer; Tan et al. is the original
  corpus (3,051 conversations, 293,297 utterances) this project draws from.
- Note for accuracy: the corpus also carries `pair_ids`/`success`/`train`
  metadata from Tan et al.'s own (unrelated) persuasion-prediction study —
  this project does not use that metadata; worth one sentence so a reader
  doesn't think you're conflating tasks.

**Synthesis paragraph to write (not listed facts — this is where your own
sentence-writing goes):** tie all of the above into one explicit gap
statement: existing fallacy detection, across both curated-benchmark
(MAFALDA) and domain-specific (Goffredo et al.) traditions, is span-level by
construction; existing discourse/conversational toolkits (ConvoKit) are not
organized around claim pairs; this project's contribution is treating the
claim pair, linked by discourse context, as the unit of both annotation and
prediction.

---

## 6. Dataset and Data Construction — verified pipeline numbers

| Stage | Count |
|---|---|
| Raw CMV utterances (ConvoKit, Tan et al. 2016 corpus) | 293,297 (3,051 conversations) |
| Reply-structure candidates (motte-bailey shape + strawman shape) | 376,714 (119,452 motte-bailey + 257,262 strawman) |
| Keyword-filtered candidates | 5,078 (3,109 motte-bailey-shaped, 1,969 strawman-shaped) |
| Sampled for manual claim extraction (6 batches) | 300 |
| Usable after claim extraction (`OK` status) | 240 |
| Hand-labeled, real (3 of 6 batches) | 91 |
| + synthetic (hand-written) | 20 |
| **Final dataset** | **111** |

**Candidate generation** (describe `src/extraction/build_candidates.py`
precisely): walks each conversation's reply tree. Motte-bailey shape: A →
(different-speaker challenge, direct child) → (same-speaker-as-A reply, child
of challenge). Strawman shape: A → (different-speaker direct reply).

**State as an explicit empirical finding, not an aside**: reply structure
alone is a very weak filter — 376,714 raw candidates is effectively "most
conversational turns," confirming that structural position (who replied to
whom) doesn't distinguish fallacious from ordinary exchanges. This motivates
the keyword filter.

**Keyword filter** (`src/extraction/filter_candidates.py`): candidate B-side
text must contain one of a fixed phrase list — "I never said," "I'm not
saying," "I didn't say," "to be clear," etc. (motte-bailey list, 15 phrases)
or "so you're saying," "are you saying," "what you're saying is," etc.
(strawman list, 12 phrases). Reduced pool to 5,078.

**State the selection-bias limitation explicitly here (not only in §13)**:
this filter makes annotation tractable but guarantees every candidate
contains a specific lexical marker, which later (§11/§12) is shown to be a
feature both models over-rely on. This is a direct, traceable link between a
data-construction decision and a downstream failure mode — worth stating as
exactly that link.

**Claim extraction**: manual distillation of raw (500–3,000 word) CMV
comments into clean claim_A/challenge/claim_B triples, with three outcomes:
`OK` (clean single claim pair), `SPRAWLING_SKIP` (comment spans multiple
unrelated topics — do not force a summary), `LOW_CONFIDENCE` (structurally
unclear/mismatched roles). Across 6 batches: 240 of 300 (80%) usable.

---

## 7. Annotation Protocol

**Labels**: `MOTTE_BAILEY`, `STRAWMAN`, `NONE`, `UNCERTAIN`.

**Core rule (state this precisely — it's a real methodological choice)**:
annotation is performed using *only* the extracted claim_A / challenge /
claim_B text, not the full original thread. If the correct label depends on
context not present in those three texts (e.g., a challenge references
something claim_A's speaker said in an earlier, un-extracted turn), the
example is marked `UNCERTAIN` rather than guessed. `UNCERTAIN` examples are
excluded from training and evaluation entirely.

**NONE vs. UNCERTAIN (state the distinction precisely, this was a specific
design decision)**: `NONE` means the three texts provide *sufficient evidence*
to conclude no fallacy occurred — e.g., an honest clarification, a fair
paraphrase, a claim restated without narrowing. `UNCERTAIN` means the
annotator cannot tell either way from what's visible.

**Real worked examples to cite** (from `ANNOTATION_GUIDELINE.md`, verbatim,
do not alter):
- Motte-bailey positive example: given in §2 above (t3_2rnfn0).
- Motte-bailey counter-example (explicitly labeled NONE in the guideline, not
  MOTTE_BAILEY): a person clarifying "I meant we should reduce *unnecessary*
  car usage, not ban cars," on their *first* statement (not a retreat from a
  stronger claim they're still leaning on) — the guideline explicitly
  contrasts this with the positive example to show where the line is drawn.
- Strawman guidance example (from the guideline): A says "we should moderately
  increase immigration," B says "so you want open borders?" — a distortion to
  a more extreme position than what was stated, vs. the counter-example
  "so, are you saying allowing guns on campus would likely be beneficial?"
  which the guideline explicitly flags as reading like a genuine clarifying
  question, not a deliberate distortion ("lean NONE or UNCERTAIN unless claim_B
  clearly attacks a stronger/dumber version of claim_A than what was actually
  said").

**Synthetic data**: 20 examples hand-written by the annotator (not LLM-
generated), following the same four-label scheme, added to reach a more
usable class balance faster than additional real-data mining would allow in
the available time. **State explicitly and without hedging**: these are not
naturally occurring CMV examples; they are included as a separate, tagged
(`source` column) subset specifically so that real-vs-synthetic comparisons
remain possible (see §11.3), not presented as equivalent in kind to the
mined data.

**Final composition**:

| Label | Real (CMV) | Synthetic | Total |
|---|---|---|---|
| NONE | 48 | 6 | 54 |
| MOTTE_BAILEY | 20 | 8 | 28 |
| STRAWMAN | 23 | 6 | 29 |
| **Total** | **91** | **20** | **111** |

(By shape: 58 motte-bailey-shaped rows, 53 strawman-shaped rows, across both
sources.)

---

## 8. Methodology

**NLI features** (`src/features/nli_features.py`): frozen, pretrained
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (Laurer et al.
2024). For each pair, score both directions:

$$P_{\text{entail}}(A\!\to\!B),\ P_{\text{neutral}}(A\!\to\!B),\ P_{\text{contra}}(A\!\to\!B)$$
$$P_{\text{entail}}(B\!\to\!A),\ P_{\text{neutral}}(B\!\to\!A),\ P_{\text{contra}}(B\!\to\!A)$$

Six features total. **State explicitly: the NLI model is used only as a
frozen feature extractor; it is never fine-tuned on this project's data.**

**Specificity features** (`src/features/specificity_features.py`) — describe
accurately as simple lexicon-based heuristics, not a discourse parser:
- hedge word count in A, in B, and the delta (B − A)
- hedge-rate delta (normalized by claim length)
- absolute/universal-word count delta (e.g., "all," "never," "always" vs.
  "some," "might," "generally")
- length ratio, len(claim_B) / len(claim_A)
- binary indicator: does claim_B contain an explicit denial phrase
  ("I never said," "I'm not saying," etc.) — `explicit_denial_B`

**Classifier**: XGBoost (gradient-boosted decision trees), scikit-learn-style
API. Hyperparameters used (state exactly, do not round or embellish):
`n_estimators=50, max_depth=3, learning_rate=0.1`, fixed `random_state=42`.
**This is not a neural model** — state this plainly if comparing to the
zero-shot LLM, to avoid implying a deep-learning-vs-deep-learning comparison.

**Per-shape binary framing**: see §3 — restate briefly here as the modeling
consequence (two independently trained/evaluated binary XGBoost models, one
per shape, rather than one 3-way model).

---

## 9. Experimental Setup

- **Evaluation protocol**: stratified 5-fold cross-validation, per shape,
  rather than a single train/test split — justified explicitly by sample
  size (44–58 examples per shape; a single split would be too noisy to trust,
  cross-validation averages over several).
- **Metrics**: accuracy, precision/recall/F1 on the positive class
  (MOTTE_BAILEY or STRAWMAN respectively), confusion matrix.
- **Datasets evaluated**: (a) 91 real-only examples, (b) 111 real+synthetic —
  reported side by side (§10.2), not merged into one number.
- **Zero-shot LLM baseline**: model used is **Claude Sonnet 5** (model ID
  `claude-sonnet-5`, Anthropic) — name it explicitly in the report, "a
  zero-shot LLM" alone is not specific enough. Given only claim_A / challenge
  / claim_B (no structural/discourse features, no training), predicts the
  per-shape binary label. Run on the 91 real examples only (not the
  synthetic 20).
  **Required limitation to state explicitly, do not omit**: Claude Sonnet 5
  also performed the manual claim extraction for this dataset, so this is not
  a fully independent, arms-length zero-shot evaluation — residual
  familiarity with specific examples cannot be ruled out. Name this as a
  threat to validity for this specific number, not a footnote.

---

## 10. Results

**Table 1 — Dataset statistics** (data already given in §6/§7 tables above —
reuse as one combined LaTeX table with columns: Stage, Count, and a second
table for label × source).

**Table 2 — XGBoost vs. zero-shot LLM (91 real examples)**

| Shape | Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Motte-bailey | XGBoost (NLI+specificity) | 44 | 0.636 | 0.611 | 0.550 | 0.579 |
| Motte-bailey | Zero-shot LLM | 44 | 0.773 | 0.750 | 0.750 | 0.750 |
| Strawman | XGBoost (NLI+specificity) | 47 | 0.617 | 0.619 | 0.565 | 0.591 |
| Strawman | Zero-shot LLM | 47 | 0.809 | 0.818 | 0.783 | 0.800 |

**Table 3 — Real-only vs. real+synthetic (XGBoost)**

| Shape | Data | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Motte-bailey | 91 real | 44 | 0.636 | 0.611 | 0.550 | 0.579 |
| Motte-bailey | 111 real+synthetic | 58 | 0.569 | 0.545 | 0.643 | 0.590 |
| Strawman | 91 real | 47 | 0.617 | 0.619 | 0.565 | 0.591 |
| Strawman | 111 real+synthetic | 53 | 0.547 | 0.586 | 0.586 | 0.586 |

**One-line interpretation to write per table** (facts only, your sentences):
zero-shot LLM clearly outperforms the engineered-feature XGBoost model on
both shapes on the real data (Table 2); adding synthetic data did not improve
XGBoost and mildly reduced it on both shapes (Table 3) — mechanism examined
in §11.3.

**Figure placement**: `figures/model_comparison.png` belongs here, after
Table 2, as the visual counterpart.

---

## 11. Ablation and Analysis

### 11.1 Feature ablation

**Table 4 — Feature ablation (XGBoost, 111 real+synthetic, 5-fold CV)**

| Shape | Feature set | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Motte-bailey | NLI only | 58 | 0.724 | 0.731 | 0.679 | 0.704 |
| Motte-bailey | Specificity only | 58 | 0.500 | 0.483 | 0.500 | 0.491 |
| Motte-bailey | NLI + Specificity | 58 | 0.569 | 0.545 | 0.643 | 0.590 |
| Strawman | NLI only | 53 | 0.566 | 0.594 | 0.655 | 0.623 |
| Strawman | Specificity only | 53 | 0.528 | 0.571 | 0.552 | 0.561 |
| Strawman | NLI + Specificity | 53 | 0.547 | 0.586 | 0.586 | 0.586 |

**Interpretation (facts, not your final wording)**: for motte-bailey, NLI-
only (0.724 accuracy) clearly outperforms the combined feature set (0.569) —
adding specificity features *reduces* performance rather than complementing
the NLI signal. Same direction, smaller magnitude, for strawman on F1 (0.623
NLI-only vs. 0.586 combined), though accuracy is closer (0.566 vs. 0.547).
Correct interpretation to state: on this dataset size, the simple lexical
specificity features appear to add noise to the combined feature vector
rather than complementary signal — **do not generalize this to "specificity
features are useless"**; state it as dataset-size-and-feature-engineering-
specific, since the specificity features are also the simplest (5 hand-built
numbers vs. 6 NLI-model-derived ones) and a larger dataset or better-designed
lexical features might behave differently.

### 11.2 NLI feature-space analysis

Figures: `nli_scatter.png` (entail(A→B) vs. entail(B→A), colored by label),
`nli_asymmetry_by_label.png` (histogram of entail(A→B) − entail(B→A), one
panel per label).

**State this as a genuine null result, precisely**: neither the scatter nor
the asymmetry histograms show a clean visual separation between NONE,
MOTTE_BAILEY, and STRAWMAN — the majority of points across all three labels
cluster near (0, 0) (mutual non-entailment), and the asymmetry distributions
for all three labels are concentrated near zero with long, overlapping tails.
**Correct interpretation, state explicitly**: this does not contradict the
ablation result (§11.1) that NLI features are useful — a multivariate
classifier (XGBoost, using all 6 NLI dimensions including neutral and
contradiction probabilities, not just the 2D entailment pair plotted) can
exploit combinations not visible in a 2D projection. The visualization shows
that the *simple, reducible* version of the NLI signal (bidirectional
entailment alone, or their raw difference) is not sufficient for a human-
legible decision rule — the useful signal is more distributed across the
6-dimensional space than a simple asymmetry score suggests.

### 11.3 Synthetic data analysis

Figure: `real_vs_synthetic_features.png` (two panels: denial-phrase rate bar
chart; claim-length-ratio box plot).

**Measured feature differences, real vs. synthetic** (verified numbers):

| Feature | Real (n=91) | Synthetic (n=20) |
|---|---|---|
| `explicit_denial_B` rate | 0.374 (37.4%) | 0.000 (0%) |
| entail(A→B), mean | 0.100 | 0.004 |
| entail(B→A), mean | 0.115 | 0.014 |
| NLI asymmetry, mean | −0.015 | −0.010 |
| length ratio B/A, mean | 1.218 | 0.818 |

**State the causal claim correctly — this is explicitly required**: the real
examples were selected via keyword filtering on exactly these denial/
retreat phrases (§6); the synthetic examples were hand-written without that
same trigger-based selection constraint. This is a plausible, well-evidenced
explanation **consistent with** the Table 3 performance drop — not a proven
causal mechanism. Use wording close to: "the distributions suggest a
plausible distributional mismatch between the two sources, offering one
explanation for the modest performance drop in Table 3, though a dataset of
this size cannot establish this as causal." Note also that the NLI-feature
distributions (asymmetry) do *not* differ as starkly as the denial-phrase
rate — so the mismatch is localized to the lexical features, not NLI, which
is additional supporting detail, not a separate claim.

---

## 12. Error Analysis

Source: `ERROR_ANALYSIS.md` in full — use it directly, it already contains
verified categories and quotes. Structure:

- 48 of 91 real examples misclassified by at least one model; 9 by both.
- Four categories (reproduce the table from `ERROR_ANALYSIS.md` as Table 5):
  lexical cue triggered without real retreat/distortion (both models); real
  retreat/distortion missed when not phrased with an obvious trigger (both);
  strawman embedded in an otherwise substantive rebuttal, missed (both);
  clarifying question/fair paraphrase misread as distortion (both).
- **Central interpretive sentence, state close to this wording**: detecting
  the *presence* of a linguistic marker associated with a rhetorical move
  (the denial/retreat phrase) is measurably easier for both model classes
  than determining whether the underlying rhetorical move — an actual
  narrowing of claim, or an actual distortion — occurred. This is the
  project's central empirical finding, more than either raw F1 number.
- **Hedge correctly**: call this a "suspected lexical shortcut, supported by
  qualitative error analysis," not a proven mechanism — you have 9
  qualitative examples, not a controlled intervention (that's §15's planned
  trigger-phrase-masking experiment).
- Representative quotes to use verbatim (already verified, from
  `ERROR_ANALYSIS.md`): `m026` ("I'm fully aware of how the word is used. All
  I'm saying is that usage is often incorrect" — false positive, restates
  rather than narrows); `s013` (claim_B is claim_A's exact argument with
  every instance of "car(s)" replaced by "gun(s)" — missed by the LLM, no
  trigger phrase present); `s051` (one fair correction immediately followed
  by one genuine strawman in the same reply — missed by both).

---

## 13. Discussion

This section must **synthesize**, not repeat §10–12. Below is the evidence-
to-claim mapping for the nine-point story — write it as connected prose
reasoning from result to result, not a restated list:

1. The pipeline successfully operationalizes a relational fallacy task end to
   end (dataset + two baselines + ablation + error analysis exist; this
   itself is the primary "progress" claim for a mid-submission).
2. Reply-structure alone is insufficient for candidate identification
   (376,714 from structure alone; evidence: §6).
3. Keyword filtering buys tractability (5,078 candidates, 111 labeled) at the
   cost of a traceable lexical selection bias (evidence: §6, §9).
4. XGBoost achieves modest-but-above-chance performance (F1 0.579–0.591 on
   real data; majority-class baseline would be ≈0.51–0.55 — compute and
   state this chance baseline explicitly as a point of comparison, it isn't
   in any table yet and should be).
5. Zero-shot LLM substantially outperforms XGBoost on the same real data
   (evidence: Table 2) — **but** state the non-independence limitation (§9)
   in the same breath, not just in §14, since it directly qualifies how much
   weight this result can bear.
6. NLI carries more signal alone than combined with specificity features
   (evidence: Table 4) — tentative interpretation about feature noise vs.
   dataset size, not a general claim about NLI superiority.
7. Synthetic data does not clearly help, and a plausible, evidenced
   explanation (denial-phrase rate mismatch) exists without being provably
   causal (evidence: §11.3).
8. Both model classes share a lexical-shortcut vulnerability (evidence: §12)
   — this is the strongest single finding in the report, because it holds
   across two very different model types.
9. Therefore (this is the forward-looking synthesis sentence): before scaling
   up model complexity (e.g., fine-tuning a transformer), the next highest-
   value step is testing whether current results survive removal of the
   lexical shortcut (§15), and expanding real annotated data, since both
   current models may be rewarding surface pattern-matching over genuine
   relational reasoning.

---

## 14. Limitations and Future Work

**Limitations (state plainly, no apologetic tone — frame each as motivating
a specific next step, per the mapping below)**:

- Only 91 real labeled examples currently — small by any standard; results
  in Tables 2-4 should be read as indicative, not conclusive.
- Single human annotator; no inter-annotator agreement computed yet (directly
  relevant given Ruggeri et al. 2025's finding that fallacy judgments carry
  genuine human disagreement — this project cannot yet distinguish
  "disagreement" from "annotator error").
- 20 synthetic examples are not naturally occurring data; §11.3 shows a
  measurable distributional difference from the real data on at least one
  feature.
- Keyword-based candidate selection creates a traceable selection bias
  (§6, §9) plausibly connected to the shared shortcut found in §12.
- Zero-shot baseline is not fully independent of the extraction process
  (§9) — a threat to how much the 0.750/0.800 F1 numbers can be trusted as
  an arms-length LLM capability measurement.
- CMV is one specific discourse register (structured, good-faith-norm-driven
  online debate); findings may not transfer to other discourse types (e.g.,
  political debate transcripts, the second source your original proposal
  named but did not yet use).
- Only a frozen, off-the-shelf NLI model and simple lexicon-based features
  are used — no fine-tuned transformer yet (deliberately deferred — state the
  reason precisely: with 44-58 examples per shape, fine-tuning risks
  memorization over generalization, and a single evaluation would be too
  noisy to interpret; this is a scope decision, not an oversight).
- Per-shape binary evaluation (§3) is a deliberate, justified design choice,
  but it does mean the current work does not yet test joint 3-way
  discrimination under a design that avoids the structural-shortcut problem
  — an open methodological question for future work (e.g., features that
  vary *within* a shape, not just between shapes).
- Distinguishing legitimate claim refinement from motte-and-bailey, and
  distinguishing a fair paraphrase/clarifying question from strawman, remain
  genuinely hard even for the human annotator in some cases (cite specific
  `UNCERTAIN`-flagged or borderline examples from the guideline if you want
  a concrete anchor — e.g., the guideline's own worked counter-examples).

**Future work / remaining timeline (Oct 3 – Oct 31)** — build a dated table
from this list, do not leave it as prose-free bullets in the final paper:

- **Trigger-phrase-masking ablation** (highest priority, ~2-3 days): mask or
  remove the exact denial/retreat phrases from claim_B and re-run both
  XGBoost and the zero-shot baseline. Directly tests whether §12's finding
  reflects a real shortcut (performance should drop substantially if so) or
  whether the phrase correlates with, but isn't solely responsible for, the
  current results.
- **Additional real annotation** (~1 week): batches 4–6 already have claim
  extraction completed (149 additional usable candidates beyond the 91
  currently labeled) — hand-labeling these would roughly double the real
  dataset without any new pipeline work.
- **Inter-annotator agreement** (~2-3 days): either recruit a second human
  rater for a subsample, or run the previously-discussed LLM-as-second-rater
  proxy, explicitly flagged in the report as a proxy, not equivalent to human
  IAA.
- **Transformer fine-tuning** (~1 week, after the above): DeBERTa/RoBERTa on
  claim_A + claim_B (+ discourse features where applicable), once the dataset
  is larger — explicitly conditional on the annotation-expansion step above,
  not attempted on the current 111 examples.
- **Full ablation + final error analysis + write-up** (remaining time to
  Oct 31): repeat all current analyses on the expanded dataset, finalize.

---

## 15. Conclusion — facts to restate, briefly

- Relational fallacy detection was formalized as cross-turn claim-pair
  classification and operationalized end-to-end: candidate generation →
  filtering → claim extraction → annotation → feature-based and zero-shot
  baselines → ablation → error analysis.
- The central empirical finding is not a leaderboard number but a shared
  failure mode: both a lightweight feature classifier and a zero-shot LLM
  appear to key on a surface lexical marker rather than the underlying
  relational judgment, which the project's own data-construction process
  (keyword-filtered candidate selection) plausibly explains and the planned
  trigger-phrase-masking ablation will directly test.
- The dataset, pipeline, and all code are in the public repository (give the
  URL); remaining work before the final submission is scoped and dated above.

---

## References — verify formatting, do not invent

Use exactly the bibliography from the original project proposal
(`random.string-Proposal Final.pdf`), ACL-formatted:

- Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large
  annotated corpus for learning natural language inference. *EMNLP 2015*.
- Chang, J. P., Chiam, C., Fu, L., Wang, A. Z., Zhang, J., & Danescu-
  Niculescu-Mizil, C. (2020). ConvoKit: A toolkit for the analysis of
  conversations. *SIGDIAL 2020*.
- Dung, P. M. (1995). On the acceptability of arguments and its fundamental
  role in nonmonotonic reasoning, logic programming and n-person games.
  *Artificial Intelligence*, 77(2), 321–357.
- Goffredo, P., Chaves, M., Villata, S., & Cabrio, E. (2023). Argument-based
  detection and classification of fallacies in political debates. *EMNLP
  2023*, 11101–11112.
- Goffredo, P., Haddadan, S., Vorakitphan, V., Cabrio, E., & Villata, S.
  (2022). Fallacious argument classification in political debates. *IJCAI
  2022*, 4143–4149.
- Helwe, C., Calamai, T., Paris, P.-H., Clavel, C., & Suchanek, F. (2024).
  MAFALDA: A benchmark and comprehensive study of fallacy detection and
  classification. *NAACL 2024*, 4810–4845.
- Jin, M., et al. (2024). CoCoLoFa: A dataset of news comments with common
  logical fallacies written by LLM-assisted crowds. *arXiv:2410.03457*.
- Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024). Less
  annotating, more classifying: Addressing the data scarcity issue of
  supervised machine learning with deep transfer learning and BERT-NLI.
- Pei, J., & Jurgens, D. (2023). POTATO: The portable text annotation tool.
  *EMNLP 2023: System Demonstrations*.
- Ruggeri, F., et al. (2025). Fine-grained fallacy detection with human
  label variation. *arXiv:2502.13853*.
- Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016).
  Winning arguments: Interaction dynamics and persuasion strategies in
  good-faith online discussions. *WWW 2016*.
- Toulmin, S. E. (1958). *The Uses of Argument*. Cambridge University Press.
- University of California, Santa Barbara. Debates — The American Presidency
  Project. (Only cite if you still mention this as a planned second data
  source in Future Work; it was not actually used in the current pipeline —
  **[NEEDS INPUT]**: confirm whether to mention this as unused-but-planned or
  drop it entirely, since the proposal named it but implementation only used
  CMV.)

---

## Appendix (optional, 1 page)

Good candidates if you have room:
- Full annotation guideline text (or a condensed version) as an appendix,
  referencing `ANNOTATION_GUIDELINE.md`.
- `nli_asymmetry_by_label.png` if not used in the main body.
- `dataset_funnel.png` if you end up prioritizing other figures in-body and
  want this one in the appendix instead.
- Extra error-analysis examples beyond the 3 quoted in §12 (there are 48
  total mismatches logged; the full join is reproducible from
  `dataset_final_xgboost_oof_predictions.csv` + `zeroshot_baseline_predictions.csv`).

---

## [NEEDS INPUT] — things genuinely not resolvable from the repo alone

1. Whether to mention the UC Santa Barbara presidential-debate corpus (named
   in the original proposal as a second planned data source) in the
   Introduction/Related Work as still-planned-but-not-yet-used, or omit it
   since this mid-submission only implements the CMV pipeline. Your call —
   affects one or two sentences in §1/§5/§15.
2. A chance/majority-class baseline number is referenced in §13 point 4 above
   but was not in any saved result file — compute it directly from the label
   counts in §6/§7 (e.g., motte-bailey majority class = NONE at 30/58 = 0.517
   accuracy for the 111-set, or 24/44 = 0.545 for the 91-real set) before
   citing it, and state which denominator you used.
3. No `README.md` currently exists in the repo (your original request
   referenced one) — if the mid-submission guidelines require one, it needs
   to be written separately; it is not part of this content dossier.
