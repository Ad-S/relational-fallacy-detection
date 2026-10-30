# 11. Limitations and future work

(For the separate section placed just before the references; it must not contain new results.)

## Limitations
- **Data size.** 188 real labeled pairs (about 100 per task on originals). Learning curves still rise at 100%,
  and results on the originals alone are noisy.
- **Single annotator.** No inter-annotator agreement; borderline cases (legitimate clarification vs retreat,
  pointed question vs distortion) are inherently subjective. About 20% of annotated rows were UNCERTAIN.
- **Labels on condensed claims.** Annotation was done on claims extracted from long comments; extraction can
  remove or add nuance, and some judgments need context beyond the extracted pair.
- **Selection bias from trigger phrases.** Every real candidate contains a denial or attribution phrase in B.
  Fallacies expressed without these phrases are absent, results describe trigger-bearing exchanges, and the
  motte-and-bailey model partly relies on the phrases (masking experiment).
- **Generated variants.** 475 of 683 pairs are generated; their labels were not re-checked by hand (flip
  labels are generated; other variants inherit the source label). The models score higher on variants than
  on real pairs, so aggregate numbers overstate performance on real discussions.
- **Motte-and-bailey on real data is near chance** for all trained models (AUC 0.47-0.54 on originals).
- **Lexical features** use plain substring matching (spurious matches) and were kept unchanged for comparability.
- **RoBERTa-large-MNLI** could not be evaluated (NaN predictions under bf16 training).
- **Zero-shot baselines** are single runs; Claude Opus 5.5 was evaluated only on the 111 mid-term originals.
- **Two-stage pipeline** thresholds fixed at 0.5 and untuned; extraction quality strongly affects it.
- **Domain.** r/ChangeMyView is a moderated, good-faith debate forum; results may not transfer elsewhere.

## Future work
- Annotate more real pairs, add a second annotator and measure agreement; label the remaining extracted
  batches (5 and 8, 81 usable pairs).
- Candidate mining without trigger phrases, for example from replies in which users call out a strawman or
  goalpost-moving, to remove the selection bias and test generalisation.
- Contrast sets with minimal edits written by hand (fair B with a trigger phrase; distortion without one).
- Combine the two-stage signal with learned models; better claim extraction for stage 1.
- Annotate rephrase function (intensification, generalisation, specification) to link the labels to
  argumentation theory.
- Evaluate on other debate data (e.g. QT30 cross-turn rephrase relations) for out-of-domain testing.
