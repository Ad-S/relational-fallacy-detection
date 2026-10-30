"""Stage-1 prompts for the two-stage pipeline: extract the proposition that claim B attributes to A's author
(strawman shape) or that B denies having claimed (motte-and-bailey shape). Stage 2 (nli_stage2.py) then checks
with the frozen NLI model whether claim A actually entails that proposition."""

SYSTEM = ("You extract propositions from short debate exchanges for a computational argumentation study. "
          "Reply with a single JSON object and nothing else.")

STRAWMAN = """Claim A was written by one person. Claim B is a reply by a DIFFERENT person.

Claim A: {claim_A}

Claim B: {claim_B}

Does Claim B attribute a position to the author of Claim A - state or imply what that person says, believes or wants (for example "so you're saying ...", "you think ...", "your argument is ...", or a rhetorical question about their view)?
If yes, write that attributed position as one short standalone sentence of the form "The author of Claim A claims that ...".
Return exactly: {{"attributes": true or false, "attributed_claim": "<sentence, or empty if none>"}}"""

MOTTE_BAILEY = """A speaker wrote Claim A. Another person challenged it (Challenge C). The SAME speaker then replied with Claim B.

Claim A: {claim_A}

Challenge C: {challenge}

Claim B: {claim_B}

Does Claim B deny, narrow or reinterpret what the speaker said before (for example "I never said ...", "I'm not saying ...", "what I meant was ...", or quietly restating a weaker version)?
If yes, write (1) the claim the speaker now says they did NOT make or mean, and (2) the position the speaker now defends - each as one short standalone sentence.
Return exactly: {{"denies": true or false, "denied_claim": "<sentence, or empty>", "defended_claim": "<sentence, or empty>"}}"""

TEMPLATES = {"strawman": STRAWMAN, "motte_bailey": MOTTE_BAILEY}
