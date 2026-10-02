# 3. Claim Extraction Process

## Why this step exists

Raw CMV comments are frequently 500–3,000-word essays covering multiple,
sometimes unrelated, arguments in a single turn (a real CMV commenter might
address five separate sub-topics in one reply). A candidate pair sampled
directly from this raw text is often not a clean "claim" in the sense the
task needs — it can be an entire multi-point essay. Claim extraction is the
manual (in this project, human-performed, following the proposal's "prompted
LLM extractor" design but executed by the annotator directly rather than via
an automated LLM call) process of distilling each sampled candidate's three
raw texts into clean, short, decontextualized claims.

## Process

For each sampled candidate, the raw `claim_A` / `challenge` / `claim_B` text
was read in full and one of three outcomes assigned:

- **`OK`**: a single, coherent claim could be cleanly extracted for each of
  the (up to three) fields. A short, clean paraphrase was written for each,
  resolving pronouns/references using context (e.g., "I never said that" →
  resolved to what the speaker is denying, specifically).
- **`SPRAWLING_SKIP`**: the comment covers multiple unrelated topics and
  cannot be reduced to one claim without fabricating a summary that isn't
  really there. These were explicitly *not* force-summarized — left blank
  with a note explaining the sprawl (e.g., "covers Trump's wealth, trade
  policy, isolationism, and welfare politics in one reply").
- **`LOW_CONFIDENCE`**: the structure is unclear, roles seem mismatched
  (e.g., the "roles reversed" pattern noted in file 02), or the exchange
  doesn't actually fit the fallacy shape it was filtered for (e.g., a
  language-barrier misunderstanding, or a full concession/delta rather than a
  retreat).

## Batch-by-batch results

Six batches of 50 candidates each were sampled (25 motte-bailey + 25 strawman
per batch, without overlap across batches, tracked via
`data/annotations/used_pair_keys.json`):

| Batch | OK | SPRAWLING_SKIP | LOW_CONFIDENCE | Total |
|---|---|---|---|---|
| 1 | 38 | 8 | 4 | 50 |
| 2 | 39 | 7 | 4 | 50 |
| 3 | 41 | 5 | 4 | 50 |
| 4 | 42 | 3 | 5 | 50 |
| 5 | 39 | 4 | 7 | 50 |
| 6 | 41 | 4 | 5 | 50 |
| **Total** | **240** | **31** | **29** | **300** |

**80% (240/300) were usable after extraction.** This is itself worth stating
as a data-quality finding: a meaningful fraction (≈10%) of keyword-filtered
candidates are still unusable due to the underlying comment's sprawling,
multi-topic nature — confirming that CMV's long-form comment style is a real
practical obstacle for this kind of claim-pair extraction, independent of the
relational-fallacy question itself.

## Recurring sprawling conversations (noted during extraction)

A small number of very long individual threads generated multiple
`SPRAWLING_SKIP` candidates each, across different batches, because the same
long conversation was sampled more than once (candidate generation samples
*pairs*, not conversations, so one giant thread can produce many candidates).
Examples noted: a Mac-vs-PC laptop price comparison thread (`t3_2fm9q9`), a
Bond-films-and-feminism debate (`t3_2p11bw`), a heaven/hell theology debate
(`t3_335imf`), a college-sports-scholarships thread (`t3_1sqkke`), and an
MRM/feminism thread (`t3_2aksc1`). **Worth noting as a limitation**: this
means a handful of long conversations are overrepresented in the *candidate
pool*, though not necessarily in the *final labeled set* since most
candidates from these threads were skipped as sprawling.

## Batches actually hand-labeled for this submission

**Only batches 1, 2, and 3 were hand-labeled** for the current dataset (91
real usable examples — see file 04). **Batches 4, 5, and 6 (149 additional
usable candidates) have completed claim extraction but have not yet been
hand-labeled** — this is explicitly scoped as near-term future work (file 10)
since it requires no further pipeline engineering, only annotation time.

## Code and file references

- Extraction was performed by directly reading and transcribing from each
  batch's raw CSV (`data/annotations/batch_N.csv`) into an extracted CSV
  (`data/annotations/batch_N_extracted.csv`), then merged with the labeling
  template via `src/annotation/merge_for_labeling.py` into
  `data/annotations/batch_N_to_label.csv` (the file actually hand-labeled).
- Sampling script: `src/annotation/prepare_batch.py` (cleans Reddit markup —
  HTML entities, blockquote lines — via `html.unescape` and regex, and
  samples with a fixed seed per batch for reproducibility).
