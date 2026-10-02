# 9. Literature Review Notes

Paper-by-paper factual backbone, from the original proposal's bibliography
(`random.string-Proposal Final.pdf`). Write connected synthesizing prose from
these facts — do not submit this as a disconnected list of summaries in the
actual report.

## Helwe et al. 2024 — MAFALDA (NAACL 2024)
- **Problem**: unify 30+ fallacy types into one shared taxonomy; benchmark
  LLMs and classifiers on fallacy detection and classification.
- **Method/dataset**: span-level annotation over curated text; evaluates
  multiple LLMs/classifiers against the unified taxonomy.
- **Relevance**: the clearest instance of the span-level paradigm this
  project departs from — a span is labeled fallacious or not, independent of
  any other span in the discourse.
- **Gap**: cannot represent a fallacy whose fallaciousness depends on a
  relationship between two individually-non-fallacious spans.

## Goffredo et al. 2022 — Fallacious Argument Classification in Political Debates (IJCAI)
- **Problem**: detect and classify fallacies in political debate transcripts.
- **Dataset**: annotated transcripts from 31 U.S. presidential debates, 1,628
  fallacious arguments.
- **Method**: Transformer-based classifiers.
- **Result/relevance**: finds that argument components and their relations
  are useful for the task — an early signal that relational structure matters
  even within a span-classification framing. Also the domain-adjacent
  precedent for political-debate transcripts as a data source (named in the
  original proposal as a second planned source, not yet implemented in this
  mid-submission — see file 10/the NEEDS-INPUT note).
- **Gap**: still predicts a fallacy label per argument/span, not per
  claim-pair relation.

## Goffredo et al. 2023 — Argument-based Detection and Classification of Fallacies in Political Debates (EMNLP)
- **Problem/method**: extends the 2022 work; explicitly represents claims,
  premises, and support/attack relations; combines Transformer
  representations with engineered features.
- **Result**: reports that adding non-text argumentative features helps
  performance.
- **Relevance**: the closest prior work in spirit — explicit argumentative
  structure plus engineered features plus a Transformer, directly echoed in
  this project's own XGBoost-with-engineered-features vs. zero-shot-LLM
  comparison (file 06).
- **Gap**: the "relation" here is support/attack between argument components
  within standard argument-mining structure, not a fallacy *defined by* a
  cross-turn relation between two claims.

## Jin et al. 2024 — CoCoLoFa (arXiv:2410.03457)
- **Problem**: scale fallacy-labeled dataset construction via LLM-assisted
  crowdsourcing, applied to news comments.
- **Method**: small human-labeled seed set → LLM-assisted generation → human
  validation → larger training set.
- **Relevance**: direct methodological precedent for this project's planned
  (and partially executed) seed-dataset-plus-augmentation approach. **Be
  precise about the difference**: this project's synthetic augmentation (file
  04) was hand-written by the annotator directly, not LLM-generated-then-
  verified — structurally similar idea (seed set + augmentation), different
  generation mechanism. Don't conflate the two.
- **Gap**: doesn't address the relational task; the fallacies covered are
  conventional single-span types.

## Ruggeri et al. 2025 — Fine-grained Fallacy Detection with Human Label Variation (arXiv:2502.13853)
- **Problem**: shows that fallacy judgments carry genuine annotator
  disagreement rather than a single objective ground truth, even for
  single-span fallacy judgments.
- **Relevance**: direct justification for (a) this project's `UNCERTAIN`
  label and the "label only from what's visible" protocol (file 04), and (b)
  treating the solo-annotator limitation honestly in the Limitations section
  rather than presenting labels as objective ground truth. Also the
  motivation named in the original proposal for an LLM-as-second-rater
  agreement check, which has been discussed but not yet run (file 10).
- **Gap**: addresses label variation for single-span fallacy judgments; this
  project extends the same concern (genuine ambiguity, not just annotator
  error) to relational judgments, which are plausibly even more
  context-dependent.

## Laurer et al. 2024 — Less Annotating, More Classifying
- **Role in this project**: the companion/methods paper for the frozen
  DeBERTa-v3-large-mnli-fever-anli-ling-wanli NLI model used as the feature
  extractor in file 05. **State plainly**: used frozen, as a feature
  extractor only; never fine-tuned on this project's data. This is explicitly
  the "deep transfer learning to address data scarcity" approach the paper's
  own title describes, applied here as reuse of someone else's trained model
  rather than this project doing its own transfer learning.

## Bowman et al. 2015 — SNLI (EMNLP)
- **Role**: defines the NLI task formally (entailment/neutral/contradiction
  given a premise and hypothesis) — cite in Background (file 01) when
  formally introducing NLI, before file 05 uses it as a feature source.

## Chang et al. 2020 — ConvoKit (SIGDIAL)
- **Role**: the toolkit providing programmatic access to the CMV corpus used
  in this project (file 02), including reply-tree traversal — this is the
  tool, not a fallacy-detection paper; cite it as infrastructure.

## Tan et al. 2016 — Winning Arguments (WWW)
- **Role**: the original paper behind the "Winning Arguments" / CMV corpus
  (3,051 conversations, 293,297 utterances) this project draws its raw data
  from. **Note for accuracy, worth a sentence in the report**: Tan et al.'s
  own study used this corpus for a different, unrelated task (predicting
  which of two matched replies to the same post was more persuasive) — the
  corpus carries leftover `pair_ids`/`success`/`train` metadata from that
  study, which this project does not use.

## Toulmin 1958 — The Uses of Argument
- **Role**: claim/evidence/warrant decomposition of argument structure —
  grounds the notion of "claim" as the principled unit to extract (file 03)
  and compare across turns, rather than comparing raw utterances directly.

## Dung 1995 — Abstract Argumentation Frameworks (Artificial Intelligence)
- **Role**: represents arguments as graph nodes with attack/support edges
  between them — the direct conceptual ancestor of this project's framing of
  a fallacy as a specific, detectable *edge type* between two claim-nodes,
  rather than a property of a single node. This is the theoretical
  justification for casting the task as claim-*pair* classification at all
  (file 01).

## Pei & Jurgens 2023 — POTATO (EMNLP System Demonstrations)
- **Role**: the annotation tool named in the original proposal for the
  planned 4-annotator team. **Note for accuracy**: not actually used in this
  solo mid-submission — annotation was performed directly in spreadsheet/CSV
  form given the single-annotator scale. Mention only if explaining the
  evolution from the original team-based plan.

## Synthesis statement to build your Related Work section's closing paragraph around

Existing fallacy detection — across both curated-benchmark (MAFALDA) and
domain-specific (Goffredo et al.) traditions — is span-level by construction:
the unit of prediction is a single utterance or argument component. Existing
discourse/conversational toolkits (ConvoKit) provide the infrastructure for
traversing multi-turn discourse but are not organized around claim pairs as
an annotation unit. This project's contribution is treating the claim pair,
linked by discourse context (a challenge, for motte-bailey; a direct reply,
for strawman), as the unit of both annotation and prediction — directly
addressing the representational gap, while explicitly building on Toulmin's
notion of a claim and Dung's edge-based view of argumentative relations for
the theoretical framing, and on Ruggeri et al.'s human-label-variation
findings for the annotation protocol itself.
