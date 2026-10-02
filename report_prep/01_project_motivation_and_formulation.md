# 1. Project Motivation and Formulation

## Title (from the original proposal)

"Detecting Cross-Turn Rhetorical Fallacies in Discourse: Motte-and-Bailey
Claim-Shift and Strawman Mischaracterization"

## The core idea

Existing computational fallacy detection (MAFALDA, Goffredo et al.) treats a
fallacy as a property of a single text span — the model is asked "does this
sentence/paragraph contain a fallacy?" This works for many fallacy types, but
breaks down for two specific ones that only exist in the *relationship*
between two statements made at different points in a conversation.

## The two fallacies, defined precisely

### Motte-and-bailey
- **Motte** = a modest, easily-defensible claim.
- **Bailey** = a bolder, more controversial claim the person actually wants
  to hold.
- The fallacy: a speaker asserts the bailey, gets challenged, retreats to the
  motte, but acts as though defending the motte also defends the bailey.
- **Why this can't be caught by single-span classification**: read alone,
  the bailey statement is just an opinion. Read alone, the motte statement is
  a reasonable, modest claim. Neither sentence is individually fallacious —
  the fallacy is the *treating-B-as-if-it-defends-A* move, which only exists
  across the pair.

### Strawman
- Person A makes a claim.
- Person B restates it as something stronger, more extreme, or simply
  different from what A said, then argues against that distorted version.
- **Why this can't be caught by single-span classification**: B's rebuttal
  might be perfectly logically valid *against the distorted claim* — the
  flaw is only visible by comparing B's characterization to A's actual words.

## The real example that grounds this (from the actual dataset)

From `data/annotations/ANNOTATION_GUIDELINE.md`, conversation `t3_2rnfn0`
(a real exchange from r/ChangeMyView about home-security policy):

> **claim_A**: "If these measures are expected and treated as reasonable,
> then it serves to make the crime more acceptable..."
>
> **challenge**: "I'm sorry, but no, this is hogwash. Leaving our doors and
> windows unlocked is not going to reduce theft, it's going to increase it."
>
> **claim_B**: "I never said that leaving them unlocked reduces theft...
> You said 'If these measures are expected and treated as reasonable, then
> it serves to make the crime more acceptable.' Essentially, that locking
> doors increases crime."

Read claim_A alone: unremarkable claim about social norms and crime.
Read claim_B alone: a reasonable clarification. Only the pair — someone
retreating to "I never said that" while still asserting the original framing
— reveals the motte-and-bailey move.

## The literature gap (stated precisely, for the Introduction/Related Work)

Two distinct problems, not one:
1. **A representational gap**: no existing fallacy dataset treats a *pair of
   claims linked by discourse context* as the unit of annotation. Existing
   corpora (MAFALDA, Goffredo et al.'s debate datasets) annotate spans.
2. **A modeling gap that follows from (1)**: you cannot fix this by training
   a bigger single-span classifier — the information needed (a second claim,
   and the discourse relationship between the two) is structurally absent
   from the single-span framing, no matter how good the classifier is.

**Hedging language to use**: "to our knowledge" when describing the gap —
not "no one has ever done this." A systematic search was not exhaustively
performed beyond the papers in the related-work review; the correct claim is
about what the *reviewed* literature does, not an absolute claim about the
entire field.

## Formal problem statement

**Input**: a candidate pair consisting of:
- `claim_A` (text)
- `challenge` (text, present only for motte-bailey-shaped candidates — a
  third-party turn between A and B)
- `claim_B` (text)

**Output**: for motte-bailey-shaped candidates, one of `{MOTTE_BAILEY, NONE}`.
For strawman-shaped candidates, one of `{STRAWMAN, NONE}`.

**Why two separate binary tasks, not one 3-way classifier** — this is a
specific, important methodological decision made partway through the project
and must be explained clearly in the report (see
`05_methodology_features_and_models.md` for the full reasoning and the
discourse-feature-constant argument).

## Research questions

1. **RQ1**: Can semantic (NLI) and lexical (specificity) features distinguish
   genuine motte-bailey/strawman pairs from superficially similar
   non-fallacious exchanges?
2. **RQ2**: How does a lightweight, task-specific feature-based classifier
   compare to a zero-shot LLM baseline on the same task?
3. **RQ3**: Of the two feature families used, which contributes more signal?
4. **RQ4**: Does adding hand-written synthetic examples to the training data
   improve performance?
5. **RQ5**: What failure modes are shared across model classes, and what do
   they reveal about the task's actual difficulty?

## Project context (for framing the mid-submission correctly)

- Originally scoped for a 4-person team; the student is now working solo
  (teammates dropped the course) — scope was deliberately reduced throughout
  (e.g., annotation target reduced from the original 150-250 to ~110-150;
  transformer fine-tuning deferred; political-debate-transcript data source
  from the original proposal not yet implemented, CMV-only for now).
- This is a **mid-submission**: dataset construction, baselines, and initial
  analysis are complete; transformer fine-tuning and larger-scale annotation
  are explicitly scoped as remaining work (see file 10).
