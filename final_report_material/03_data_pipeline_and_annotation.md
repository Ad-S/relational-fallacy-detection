# 3. Data pipeline and annotation (real r/ChangeMyView pairs)

## 3.1 Corpus
- ConvoKit "Winning Arguments" corpus of r/ChangeMyView (Tan et al., 2016), processed locally.
- 293,297 utterances, 3,051 conversations, 34,911 speakers. Persuasion metadata is not used; only the
  reply structure and text.

## 3.2 Structural candidate generation (`src/extraction/build_candidates.py`)
- Walks every conversation's reply tree (`reply_to`).
- **Strawman candidate:** A -> B, B is a direct reply to A by a different speaker.
- **Motte-and-bailey candidate:** A -> C -> B, C replies to A from a different speaker, B is A's speaker
  replying to C.
- A and B must be at least 40 characters and not `[deleted]` / `[removed]` (C is not filtered).
- Output: **119,452** motte-and-bailey-shaped and **257,262** strawman-shaped candidates (376,714 total).
  Reply structure alone cannot tell fallacious exchanges from normal conversation.

## 3.3 Trigger-phrase filter (`src/extraction/filter_candidates.py`)
Case-insensitive substring match on the full text of B.
- **Motte-and-bailey (16 phrases):** "i never said", "i didn't say", "i did not say", "what i meant",
  "i meant was", "that's not what i said", "that is not what i said", "to be clear,", "i only said",
  "i only meant", "all i said was", "all i'm saying is", "i wasn't saying", "i was not saying",
  "i'm not saying", "i am not saying".
- **Strawman (12 phrases):** "so you're saying", "so you are saying", "so basically you're saying",
  "what you're saying is", "what you are saying is", "are you saying", "you're saying that",
  "you are saying that", "you seem to be saying", "you're essentially saying", "so your argument is",
  "so what you're saying".
- Output: **3,109** motte-and-bailey + **1,969** strawman = **5,078** candidates.
- This makes annotation feasible but introduces a **selection bias**: the data only contains exchanges in
  which B uses one of these phrases (see 3.8).

## 3.4 Sampling (`src/annotation/prepare_batch.py`)
- Batches of 50 candidates (25 per shape), fixed seed per batch (42 + batch number), no candidate sampled
  twice (`used_pair_keys.json`).
- Text cleaning: HTML entities unescaped, quoted lines (starting with ">") removed, whitespace collapsed.
- 8 batches sampled (400 candidates).

## 3.5 Claim extraction
Raw CMV comments are often long, multi-topic essays. Each sampled candidate was condensed into short,
self-contained claims A, C (motte-and-bailey only) and B, resolving pronouns and references from context,
with an LLM (Claude Sonnet 5). Each candidate received one status:
- **OK:** a single coherent claim could be extracted for every field.
- **SPRAWLING_SKIP:** the comment covers several unrelated topics and cannot be reduced to one claim.
- **LOW_CONFIDENCE:** unclear structure, mismatched roles, or the exchange does not fit the shape.

| Batch | OK | SPRAWLING_SKIP | LOW_CONFIDENCE | Annotated? |
|---|---|---|---|---|
| 1 | 38 | 8 | 4 | yes |
| 2 | 39 | 7 | 4 | yes |
| 3 | 41 | 5 | 4 | yes |
| 4 | 42 | 3 | 5 | yes |
| 5 | 39 | 4 | 7 | no |
| 6 | 41 | 4 | 5 | yes |
| 7 | 35 | 9 | 6 | yes |
| 8 | 42 | 3 | 5 | no |
| **Total** | **317** | **43** | **40** | 6 of 8 batches |

79% of sampled candidates were usable after extraction. Median claim length in the final data: about
30 words for A, 25 for C, 32 for B.

## 3.6 Annotation (`data/annotations/ANNOTATION_GUIDELINE.md`)
- One annotator labeled every OK row of batches 1, 2, 3, 4, 6 and 7 (236 rows) from the extracted claims.
- Labels: MOTTE_BAILEY, STRAWMAN, NONE, UNCERTAIN.
- **Motte-and-bailey rule:** is B a narrower or weaker version of A, offered as if it still defends A?
  Honest clarification or added detail is NONE.
- **Strawman rule:** does B distort A into something stronger or different and argue against that?
  An honest attempt to understand A (even a pointed question) or a fair paraphrase is NONE.
- **UNCERTAIN:** cannot be decided from the extracted claims (for example, missing context). Excluded.
- Guideline example (motte-and-bailey, thread t3_2rnfn0): A "If these measures are expected and treated as
  reasonable, then it serves to make the crime more acceptable..."; C "...this is hogwash. Leaving my doors
  and windows unlocked is not going to reduce theft, it's going to increase it."; B "I never said that
  leaving them unlocked reduces theft... Essentially, that locking doors increases crime."

## 3.7 Consolidation (`src/annotation/consolidate_v2.py`)
- Labels are keyed on pair_id and normalised: case-insensitive; `not_motte_bailey` / `not_strawman` /
  `none` -> NONE; `uncertain` and `insufficient_context` -> dropped. A label belonging to the other shape
  stops the script. Rows whose extraction status is not OK are dropped regardless of label.
- Claim text is checked against the original extraction (identical apart from whitespace and a leading
  apostrophe stripped by a spreadsheet in 4 rows).
- A row-by-row audit is written to `dataset_consolidated_v2_label_audit.csv`.

| Outcome of the 300 annotated-batch rows | Rows |
|---|---|
| Kept (labeled) | **188** |
| UNCERTAIN / insufficient context | 47 |
| SPRAWLING_SKIP | 36 |
| LOW_CONFIDENCE | 28 |
| OK row with missing claim text | 1 |

Kept rows per batch:

| Batch | MB-shaped: MOTTE_BAILEY | MB-shaped: NONE | SM-shaped: NONE | SM-shaped: STRAWMAN | Total |
|---|---|---|---|---|---|
| 1 | 8 | 6 | 9 | 8 | 31 |
| 2 | 4 | 10 | 8 | 8 | 30 |
| 3 | 8 | 8 | 7 | 7 | 30 |
| 4 | 4 | 10 | 14 | 6 | 34 |
| 6 | 0 | 15 | 8 | 7 | 30 |
| 7 | 3 | 14 | 2 | 14 | 33 |
| **Total** | **27** | **63** | **48** | **50** | **188** |

- The 188 real pairs come from 169 conversations; 35 rows share a conversation with another row.
- The first 91 of these rows (batches 1-3) are identical to the mid-term dataset (labels and text verified).
- About 20% of annotated OK rows were UNCERTAIN, a measure of how hard the task is from claims alone.

## 3.8 Selection bias: trigger phrases in the labeled data
Share of rows whose extracted claim B still contains one of the selection phrases:

| Source | MB-shaped: MOTTE_BAILEY | MB-shaped: NONE | SM-shaped: STRAWMAN | SM-shaped: NONE |
|---|---|---|---|---|
| Real | 26/27 (96%) | 56/63 (89%) | 38/50 (76%) | 34/48 (71%) |
| Hand-written | 0/8 | 0/6 | 0/6 | - |
| Generated variants | 52/92 (57%) | 84/157 (54%) | 31/124 (25%) | 38/102 (37%) |

In the real data the phrase is present in nearly every example of **both** classes, so it cannot
separate them on its own - but a model can still associate the phrase with a fallacy. The trigger-masking
experiment (`08_analysis_what_models_use.md`) tests this directly.
