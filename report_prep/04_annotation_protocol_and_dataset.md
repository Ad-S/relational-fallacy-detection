# 4. Annotation Protocol and Final Dataset

## Labels

Four labels, defined in full in `data/annotations/ANNOTATION_GUIDELINE.md`:

- `MOTTE_BAILEY`
- `STRAWMAN`
- `NONE` — confident that no relational fallacy is present
- `UNCERTAIN` — genuinely cannot tell; **excluded from training/evaluation**

## Core protocol rule: label from the extracted triple only

Annotation uses *only* `claim_A`, `challenge` (where applicable), and
`claim_B` — not the full original conversation thread. This was a deliberate,
explicit decision (matching what the eventual model will also see). The
direct consequence: **if the correct label depends on context outside those
three texts (e.g., a challenge refers to something claim_A's speaker said in
an earlier, unextracted turn), the example is marked `UNCERTAIN` rather than
guessed.**

**Important sub-rule, decided during annotation**: if claim_B itself contains
enough information to resolve an apparent mismatch (e.g., a sentence like "I'm
not the person who made the original comment, I'm not defending it"), that
counts as resolvable from the visible text — use `NONE` or the fallacy label
as appropriate, not `UNCERTAIN`. `UNCERTAIN` is reserved specifically for when
the resolution genuinely isn't present anywhere in the three texts.

## NONE vs. UNCERTAIN — the precise distinction

- **`NONE`**: the three texts provide *sufficient evidence* to confidently
  conclude no fallacy occurred (e.g., an honest clarification, a fair
  paraphrase, a claim restated without any narrowing, a scope-limiting
  response that doesn't retreat from anything).
- **`UNCERTAIN`**: the annotator cannot tell either way from what's visible —
  used when the text is genuinely ambiguous, not as a default when unsure for
  a moment.

**Practical heuristic used**: "if you can explain in one sentence why it's
fine, call it NONE; if you keep going back and forth, call it UNCERTAIN and
move on."

## Worked examples from the actual guideline (verbatim, not invented)

**Motte-bailey positive example** (label `MOTTE_BAILEY`, conversation
`t3_2rnfn0`): see file 01 for the full quote — "I never said that leaving
them unlocked reduces theft..." retreat.

**Motte-bailey counter-example** (the guideline explicitly contrasts this to
show where the line is): a hypothetical where someone says "we should reduce
car usage" and, when pushed, says on their *first* statement "I meant we
should reduce *unnecessary* car usage, not ban cars" — if this is their only
statement (not a retreat from a stronger claim they're still relying on), the
guideline calls this `NONE`, normal clarification, not motte-and-bailey.

**Strawman guidance** (from the guideline): shape is `claim_A` (original
claim) → `claim_B` (someone else's reply, restating/interpreting claim_A).
**Ask**: does claim_B *distort* claim_A into something stronger/different,
and argue against that? Worked example: A says "we should moderately
increase immigration," B says "so you want completely open borders?" — the
guideline says this counts as STRAWMAN. **Explicit counter-example from the
guideline, marked as borderline/likely NONE**: a real example found in the
data where A says "most mass shootings are where carrying is banned," and B
asks "so, are you saying allowing guns on campus would likely be beneficial?"
— the guideline explicitly says: "this reads like a genuine clarifying
question, not a deliberate distortion. Lean NONE or UNCERTAIN unless claim_B
clearly attacks a stronger/dumber version of claim_A than what was actually
said."

## A specific, real methodological discussion that shaped the protocol

During annotation, a borderline case came up (CMV conversation about ACA/
healthcare funding, pair `m021`): claim_A argues the wealthy have some tax
obligation to fund healthcare; the challenge changes the subject to "the real
problem is government spending badly, not lack of money"; claim_B replies
"that's a different line of reasoning... outside the scope of what I'm
addressing." This was correctly labeled `NONE` — the challenge doesn't
restate claim_A as something distorted (so it isn't evidence of a strawman
by the challenger within this row's scope), and claim_B doesn't retreat from
anything (so it isn't a motte-bailey either) — it's simply a scope-limiting
reply to an off-topic objection. **General rule this established**: a
challenge "changing the subject" rather than distorting the claim does not,
by itself, justify a fallacy label; the row's label describes only the
specific (claim_A, claim_B) relationship it was generated to capture.

## Synthetic data

**20 examples were hand-written** (by the annotator, not LLM-generated) to
supplement the real-data annotation, following the same four-label scheme and
the same two shapes (motte-bailey: claim_A/challenge/claim_B triples;
strawman: claim_A/claim_B pairs). Added specifically to reach a usable class
balance faster than additional real-data mining would allow in the available
time, inspired by the proposal's own planned "LLM-assisted augmentation"
methodology (Jin et al. 2024, CoCoLoFa) but executed by direct human writing
rather than LLM generation plus verification.

**These are explicitly not presented as equivalent to naturally occurring CMV
data.** They are tagged with a `source` column (`reddit_cmv` vs.
`synthetic_handwritten`) specifically so real-vs-synthetic comparisons remain
possible — see file 07 for the finding that they differ measurably on at
least one feature (`explicit_denial_B` rate: 37.4% real vs. 0.0% synthetic).

**A quality note from reviewing the synthetic examples**: three of the
hand-written strawman examples (numbered 11, 14, 17 in the original batch)
were flagged as weaker than the others during review — they read more like
direct counter-arguments/counter-examples than genuine distortions of the
original claim (e.g., one gives a counter-example about wearing a vest to a
wedding rather than restating the original "personal style is subjective"
claim as something more extreme). These were kept in the dataset as labeled
but are worth a specific mention in Limitations if there's room, since they
represent lower annotation confidence even within the synthetic subset.

## Final dataset

File: `data/annotations/dataset_final.csv` — **111 examples total.**

**By label:**

| Label | Real (CMV) | Synthetic | Total |
|---|---|---|---|
| NONE | 48 | 6 | 54 |
| MOTTE_BAILEY | 20 | 8 | 28 |
| STRAWMAN | 23 | 6 | 29 |
| **Total** | **91** | **20** | **111** |

**By shape** (`type_candidate` column): 58 motte-bailey-shaped rows, 53
strawman-shaped rows.

**UNCERTAIN examples excluded**: across the 3 hand-labeled batches (150
candidates), 27 were marked `UNCERTAIN` and excluded from the final dataset
(not counted in the 91). One label typo (`"MAYBE NONE"` on row `m017`) was
found during dataset consolidation and corrected to `NONE` after review —
the underlying example (a race/prejudice discussion where claim_B reasserts
the same position with a new supporting example, no retreat) clearly fit
`NONE`'s definition.

## Code and file references

- Guideline: `data/annotations/ANNOTATION_GUIDELINE.md`
- Per-batch labeled files: `data/annotations/annotated_batch_{1,2,3}_to_label.csv`
- Consolidation script (merges batches, normalizes labels, drops UNCERTAIN):
  `src/annotation/consolidate_dataset.py` → produces `data/annotations/dataset_consolidated.csv` (91 real examples)
- Synthetic examples: `data/annotations/synthetic_examples.csv`
- Final merged file: `data/annotations/dataset_final.csv` (111 examples, used for all experiments in files 06-07 except where a table explicitly says "91 real")
