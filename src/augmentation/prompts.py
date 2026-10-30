"""Prompt for generating variants of hand-written examples. Definitions follow ANNOTATION_GUIDELINE.md."""

SYSTEM_PROMPT = ("You write training examples for a computational argumentation study. "
                 "Reply with a single JSON object and nothing else.")

DEFINITIONS = """Definitions:
- MOTTE_BAILEY: a speaker makes Claim A; another person challenges it (Challenge C); the ORIGINAL speaker replies with Claim B that retreats to a narrower, weaker or more easily defensible position while presenting it as if it still defends Claim A.
- STRAWMAN: one person makes Claim A; a DIFFERENT person replies with Claim B that misrepresents Claim A as a different, stronger or more extreme position and argues against that distorted version.
- NONE: no such move - an honest clarification, an open concession, a fair paraphrase, a genuine question, or ordinary disagreement with what was actually said."""

FLIP = {
    ("motte_bailey", "MOTTE_BAILEY"): ("NONE", "the speaker now openly concedes or honestly clarifies, without pretending the weaker point still defends Claim A"),
    ("motte_bailey", "NONE"): ("MOTTE_BAILEY", "the speaker now retreats to a weaker, more defensible claim while acting as if it still supports Claim A"),
    ("strawman", "STRAWMAN"): ("NONE", "the reply now responds to what Claim A actually says (a fair paraphrase or genuine disagreement) without distorting it"),
    ("strawman", "NONE"): ("STRAWMAN", "the reply now distorts Claim A into a stronger or more extreme position and argues against that"),
}

TEMPLATE = """{definitions}

Here is a hand-written {shape_name} example with its label:

{example}
Label: {label}

Write exactly 5 new examples and return them as JSON:
{{"variants": [{{"kind": "...", "claim_A": "...", "challenge": "...", "claim_B": "...", "label": "..."}}]}}

Variants 1-2, kind "paraphrase": the SAME exchange reworded with different words and sentence structure, same meaning, same rhetorical move. label: {label}.
Variants 3-4, kind "topic_transfer": a DIFFERENT everyday debate topic (a different one for each), with the SAME logical move as the original. label: {label}.
Variant 5, kind "flip": copy Claim A{keep_c} word for word, and minimally rewrite Claim B so that {flip_instruction}. label: {flip_label}.

Style: informal, like real Reddit comments; similar length to the original; 1-3 sentences per claim. Never name the fallacy in the text. {challenge_rule}"""


def build_prompt(shape, a, c, b, label):
    flip_label, flip_instruction = FLIP[(shape, label)]
    if shape == "motte_bailey":
        example = f"Claim A: {a}\nChallenge C: {c}\nClaim B: {b}"
        keep_c, rule = " and Challenge C", "Every variant needs a non-empty challenge."
        name = "motte-and-bailey-shaped (A, challenge C, then the same speaker's reply B)"
    else:
        example = f"Claim A: {a}\nClaim B: {b}"
        keep_c, rule = "", 'This shape has no challenge: set "challenge" to an empty string in every variant.'
        name = "strawman-shaped (A, then a different person's reply B)"
    return TEMPLATE.format(definitions=DEFINITIONS, shape_name=name, example=example, label=label,
                           keep_c=keep_c, flip_instruction=flip_instruction, flip_label=flip_label,
                           challenge_rule=rule), flip_label


# --- real CMV seeds: label-preserving variants only (no flips), so every label is still the human label ---
TEMPLATE_REAL = """{definitions}

Here is a {shape_name} example taken from a real online discussion, with its human label:

{example}
Label: {label}

Write exactly 2 new examples and return them as JSON:
{{"variants": [{{"kind": "...", "claim_A": "...", "challenge": "...", "claim_B": "...", "label": "..."}}]}}

Variant 1, kind "paraphrase": the SAME exchange reworded with different words and sentence structure, same meaning, same rhetorical move. label: {label}.
Variant 2, kind "topic_transfer": a DIFFERENT everyday debate topic with the SAME logical move as the original. label: {label}.

Style: informal, like real Reddit comments; similar length to the original; 1-3 sentences per claim. Ignore bracketed editorial notes such as "[Content note: ...]" and do not copy them. Never name the fallacy in the text. {challenge_rule}"""


def build_prompt_real(shape, a, c, b, label):
    if shape == "motte_bailey":
        example = f"Claim A: {a}\nChallenge C: {c}\nClaim B: {b}"
        rule = "Every variant needs a non-empty challenge."
        name = "motte-and-bailey-shaped (A, challenge C, then the same speaker's reply B)"
    else:
        example = f"Claim A: {a}\nClaim B: {b}"
        rule = 'This shape has no challenge: set "challenge" to an empty string in every variant.'
        name = "strawman-shaped (A, then a different person's reply B)"
    return TEMPLATE_REAL.format(definitions=DEFINITIONS, shape_name=name, example=example, label=label,
                                challenge_rule=rule), None
