"""Zero-shot prompts for the scripted LLM re-run. Definitions paraphrase the decision rules in
data/annotations/ANNOTATION_GUIDELINE.md; the guideline's worked examples are deliberately NOT
included because they are drawn from the dataset itself."""

SYSTEM_PROMPT = (
    "You are an expert annotator for a computational argumentation study. "
    "You judge short exchanges from an online debate forum. "
    "Reply with a single JSON object and nothing else."
)

MOTTE_BAILEY = """Decide whether this exchange contains a motte-and-bailey move.

Structure: a speaker makes Claim A. A different person challenges it (Challenge C). The ORIGINAL speaker replies with Claim B.

Label MOTTE_BAILEY if Claim B retreats to a narrower, weaker or more easily defensible position than Claim A, while presenting that retreat as if it still defends Claim A (treating the weaker claim as equivalent to the original one).
Label NONE if Claim B is an honest clarification, adds detail, restates the same position, or openly concedes the point. Ordinary disagreement or clarification is NONE.

Claim A: {claim_A}

Challenge C: {challenge}

Claim B: {claim_B}

Respond with exactly this JSON: {{"label": "MOTTE_BAILEY" or "NONE", "reason": "<one sentence>"}}"""

STRAWMAN = """Decide whether this exchange contains a strawman.

Structure: one person makes Claim A. A DIFFERENT person replies with Claim B.

Label STRAWMAN if Claim B misrepresents Claim A as a different, stronger or more extreme position than the one actually stated, and argues against that distorted version.
Label NONE if Claim B fairly paraphrases Claim A, asks a genuine clarifying question (even a pointed one), or disagrees with what Claim A actually says.

Claim A: {claim_A}

Claim B: {claim_B}

Respond with exactly this JSON: {{"label": "STRAWMAN" or "NONE", "reason": "<one sentence>"}}"""

TEMPLATES = {"motte_bailey": MOTTE_BAILEY, "strawman": STRAWMAN}
VALID = {"motte_bailey": {"MOTTE_BAILEY", "NONE"}, "strawman": {"STRAWMAN", "NONE"}}
