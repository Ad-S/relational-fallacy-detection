# 2. Data Pipeline and Corpus

## Source corpus

**r/ChangeMyView (CMV)**, accessed via **ConvoKit**'s "Winning Arguments"
corpus (Tan et al., 2016), downloaded once and used entirely locally — no
live Reddit scraping, no API costs. Chosen specifically because Reddit's
2023 API pricing changes closed off free bulk access to live subreddit data,
and this corpus is a pre-existing, cleaned, static snapshot that sidesteps
that problem entirely.

CMV's defining norm: the original poster (OP) states an opinion they
genuinely hold and explicitly invites others to argue them out of it. If a
reply changes the OP's mind, the OP awards it a "∆" (delta). This produces a
large amount of sustained, good-faith, multi-turn argumentative exchange —
exactly the genre where repeated-challenge-and-retreat (motte-and-bailey) and
claim-and-rebuttal (strawman) patterns are expected to occur.

**Corpus statistics** (verified via ConvoKit's `print_summary_stats()`):
- 3,051 conversations
- 293,297 utterances
- 34,911 speakers

Note: the corpus also carries `pair_ids` / `success` / `train` metadata from
Tan et al.'s own original study (which compared successful vs. unsuccessful
persuasive replies to the same OP). **This project does not use that
metadata** — it's unrelated to the relational-fallacy task and is mentioned
here only so the report doesn't appear to conflate the two studies.

## Stage 1: Candidate generation by reply-tree shape

Script: `src/extraction/build_candidates.py`

Each conversation's reply tree is walked (using the `reply_to` field to
reconstruct who replied to whom). Two structural shapes are extracted:

**Motte-bailey shape**: `A → (different-speaker challenge, direct child of A)
→ (same-speaker-as-A reply, child of the challenge)`. I.e., a three-turn
chain: original claim, pushback from someone else, then the original
speaker's own follow-up.

**Strawman shape**: `A → (different-speaker direct reply)`. A simple
two-turn chain: original claim, someone else's direct reply.

**Result**: 376,714 raw candidates total:
- 119,452 motte-bailey-shaped
- 257,262 strawman-shaped

**This is itself a reportable empirical finding, not just a pipeline
statistic**: 376,714 is an enormous fraction of the corpus's conversational
turns — it demonstrates that *reply structure alone* (who-replied-to-whom,
in what order) does not meaningfully distinguish fallacious exchanges from
ordinary conversation. "Person responds to someone who responded to them" and
"person B replies directly to person A" both turn out to just describe
normal back-and-forth discourse. This motivated Stage 2.

## Stage 2: Keyword filtering

Script: `src/extraction/filter_candidates.py`

Candidates are kept only if `claim_B`'s raw text contains at least one phrase
from a fixed list, checked via case-insensitive substring match:

**Motte-bailey trigger phrases** (checked against the reply, i.e. the
same-speaker follow-up): "i never said", "i didn't say", "i did not say",
"what i meant", "i meant was", "that's not what i said", "that is not what i
said", "to be clear,", "i only said", "i only meant", "all i said was", "all
i'm saying is", "i wasn't saying", "i was not saying", "i'm not saying", "i am
not saying" (16 phrases total).

**Strawman trigger phrases** (checked against the reply, i.e. the other
speaker's characterization): "so you're saying", "so you are saying", "so
basically you're saying", "what you're saying is", "what you are saying is",
"are you saying", "you're saying that", "you are saying that", "you seem to
be saying", "you're essentially saying", "so your argument is", "so what
you're saying" (12 phrases).

**Result**: 5,078 candidates:
- 3,109 motte-bailey-shaped
- 1,969 strawman-shaped

**This filter is the single most consequential design decision in the data
pipeline, and must be stated as such in the report**: it makes manual review
tractable (5,078 is reviewable; 376,714 is not), but it guarantees that
*every single candidate in the dataset contains one of these specific lexical
markers*. This is a selection bias baked into the data by construction. It is
directly relevant later: `07_ablation_and_deeper_analysis.md` and
`08_error_analysis.md` show that both models tested appear to over-rely on
exactly these markers — a plausible downstream consequence of this design
decision, not an unrelated finding.

## A note on an observed pipeline quirk (worth one sentence in Limitations)

During manual review, several strawman-shaped candidates were found where the
trigger phrase appears in `claim_A` rather than `claim_B` — meaning claim_A
is itself a mischaracterizing move directed at an even earlier, unseen turn,
and claim_B is the target's defensive correction. These are still usable
examples, just with the "who is distorting whom" roles reversed from the
template the filter was designed around. This was handled during manual
annotation (see file 04) by reading the actual content rather than assuming
a fixed role mapping from the column names.

## Pipeline code reference

- `src/extraction/build_candidates.py` — Stage 1 (shape-based candidate generation)
- `src/extraction/filter_candidates.py` — Stage 2 (keyword filtering)
- Both operate on the raw ConvoKit corpus stored locally at `data/raw/winning-args-corpus` (excluded from git via `.gitignore` due to size — reproducible by re-running the download, not by cloning the repo)
- Intermediate output: `data/processed/candidates_motte_filtered.jsonl` (3,109 rows), `data/processed/candidates_strawman_filtered.jsonl` (1,969 rows) — both committed to the repo
- The unfiltered 376,714-row intermediate files were too large for git (278MB/437MB) and are excluded; regeneratable by rerunning Stage 1
