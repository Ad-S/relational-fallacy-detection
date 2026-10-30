# 1. Project overview and research questions

## Title
Detecting Cross-Turn Rhetorical Fallacies in Discourse: Motte-and-Bailey Claim-Shift and Strawman Mischaracterization.

## The problem
Most computational fallacy detection decides whether a single text span is fallacious. Two common
rhetorical fallacies cannot be seen that way, because they exist only in the relation between turns:

- **Motte-and-bailey:** a speaker makes a strong claim (the bailey, claim A); someone challenges it
  (challenge C); the same speaker retreats to a weaker, more defensible claim (the motte, claim B) while
  treating the retreat as if it still defended the original claim. Neither A nor B is fallacious alone.
- **Strawman:** a respondent replies to claim A with claim B that misrepresents A as a stronger,
  different or more extreme position and argues against that distorted version. B alone can look
  like ordinary disagreement; the fallacy is the mismatch between what A said and what B attributes to A.

## Task formulation (implemented)
- An instance is a pair of short claims, A and B. Motte-and-bailey instances also include the challenge C.
- Two structural shapes, taken from the reply tree of a conversation:
  - **Motte-and-bailey shape:** A -> C -> B, where C replies to A from a different speaker and B is A's
    speaker replying to C.
  - **Strawman shape:** A -> B, where B is a direct reply to A from a different speaker.
- Two separate binary tasks: MOTTE_BAILEY vs NONE, and STRAWMAN vs NONE.
- Why not one three-way classifier: speaker identity and the presence of a challenge are fixed by the
  candidate shape, so a three-way model could separate the shapes without reading the claims. Separate
  binary tasks remove that structural shortcut.
- Grounding: claims as the unit of comparison (Toulmin, 1958); fallacy as a relation between arguments
  (Dung, 1995); strawman as a misuse of rephrase between two propositions (Visser et al.).

## Central question of the final report
**Do models detect these fallacies by reasoning about the relation between claims, or by surface cues?**
The experiments answer this from four directions: trained models (feature-based and fine-tuned), what
the models rely on (input ablation, trigger masking, counterfactual pairs), what frozen encoders
represent (layer probing), and a theory-driven pipeline (extract the attributed or denied claim, then
check it with NLI).

## Research questions (as answered by the final experiments)
- **RQ1 - Feasibility:** can cross-turn claim-pair classification detect the two fallacies above chance?
- **RQ2 - Model comparison:** how do a feature model (XGBoost on NLI + lexical features), fine-tuned
  cross-encoders and zero-shot LLMs compare?
- **RQ3 - Relational vs surface:** does the model need the full exchange (A, C, B) rather than B alone,
  and does it depend on the trigger phrases used to select candidates?
- **RQ4 - Inference knowledge:** does NLI pretraining help, and where in the encoder is the fallacy
  signal represented?
- **RQ5 - Data:** do more data and generated variants help, in particular on the original examples?
- **RQ6 - Theory-driven detection:** does an extract-then-verify pipeline built on the
  "misuse of rephrase" account work?

## Contributions (what can be claimed)
1. **An operational cross-turn formulation** of motte-and-bailey and strawman detection as two binary
   claim-pair tasks defined by conversation structure - the first empirical implementation of the
   relational view proposed in argumentation theory (Visser et al.), as far as the reviewed literature shows.
2. **A dataset** of 683 claim pairs: 188 real hand-labeled r/ChangeMyView pairs, 20 hand-written pairs
   and 475 generated variants, with a reproducible construction pipeline.
3. **A controlled comparison** of feature-based, fine-tuned and zero-shot models under one grouped
   cross-validation protocol in which every trained model sees identical folds.
4. **Diagnostics of what models use:** input ablation, trigger-phrase masking, counterfactual flip pairs,
   layer-wise probing and an originals-only experiment - showing a trigger-phrase shortcut for
   motte-and-bailey and a genuinely relational, inference-related signal for strawman.
5. **A theory-driven two-stage pipeline** (claim extraction + NLI verification), with a mostly negative
   but informative result.

What should **not** be claimed as novel on its own: using NLI features, using XGBoost, the zero-shot
comparison, or detecting strawman in general (strawman is already a label in single-text fallacy datasets).
