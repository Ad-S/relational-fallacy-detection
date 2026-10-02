# Annotation Guideline (v1)

You're looking at pairs of claims made in the same Reddit thread and deciding
if the relationship between them is a fallacy, or just normal conversation.

For each row in `batch_1.csv`, fill in the `label` column with one of:

- `MOTTE_BAILEY`
- `STRAWMAN`
- `NONE` — no relational fallacy, just ordinary disagreement/clarification
- `UNCERTAIN` — you genuinely can't tell; don't force a guess

Use the `notes` column to jot a one-line reason. Future-you will thank
present-you when writing the paper.

## Motte-and-bailey rows

Shape: `claim_A` (a bold/strong claim) → `challenge` (someone pushes back) →
`claim_B` (same person as claim_A, replying again).

**Ask:** is claim_B a *narrower/weaker* version of claim_A, being offered
*as if* it still defends claim_A? If the person is just adding detail or
honestly clarifying (not retreating from what was challenged), that's
`NONE`, not `MOTTE_BAILEY`.

Real example found in the data (label this MOTTE_BAILEY):

> **claim_A:** "If these measures are expected and treated as reasonable,
> then it serves to make the crime more acceptable..."
>
> **challenge:** "I'm sorry, but no, this is hogwash. Leaving our doors and
> windows unlocked is not going to reduce theft, it's going to increase it."
>
> **claim_B:** "I never said that leaving them unlocked reduces theft...
> You said 'If these measures are expected and treated as reasonable, then
> it serves to make the crime more acceptable.' Essentially, that locking
> doors increases crime."

Why this counts: the person retreats to "I never said X" and reframes the
conversation around a narrower claim, while still acting like their original
broader point stands.

**Counter-example (this should be NONE, not MOTTE_BAILEY):** someone says
"we should reduce car usage" and after pushback says "I meant we should
reduce *unnecessary* car usage, not ban cars" — if this is their first and
only statement (not a retreat from a stronger claim they're still leaning
on), it's just normal clarification, not a fallacy.

## Strawman rows

Shape: `claim_A` (original claim) → `claim_B` (someone else's reply,
restating/interpreting claim_A).

**Ask:** does claim_B *distort* claim_A into something stronger/different,
and then argue against that distorted version? If claim_B is an honest,
accurate attempt to understand claim_A (even if phrased as a question), or
a fair paraphrase, that's `NONE`.

Example found in the data (needs your judgment — could go either way):

> **claim_A:** "Most to all mass shootings in the US are where carrying is
> banned (for the law abiding)."
>
> **claim_B:** "So, are you saying allowing guns on campus would likely be
> beneficial?"

This is borderline — it reads like a genuine clarifying question, not a
deliberate distortion. Lean `NONE` or `UNCERTAIN` unless claim_B clearly
attacks a stronger/dumber version of claim_A than what was actually said.

**What would make it STRAWMAN:** if claim_A says "banning is correlated
with more shootings" and claim_B replies "so you want NO gun laws at all
anywhere, that's insane" — that's arguing against an extreme position
nobody stated.

## General notes

- When in doubt between NONE and UNCERTAIN: if you can articulate *why* it's
  probably fine in one sentence, call it NONE. If you keep going back and
  forth, call it UNCERTAIN and move on — don't burn time on one row.
- Read the raw comment if `claim_A`/`claim_B` feels ambiguous — the CSV
  strips reddit quote-markup for readability but keeps everything else.
- It's fine (expected, even) if most rows end up NONE. That's real signal,
  not a failure of the filter.
