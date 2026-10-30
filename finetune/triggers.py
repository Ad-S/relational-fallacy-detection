"""Trigger phrases used to select candidates (copied from src/extraction/filter_candidates.py so this
folder is self-contained). Used only to MASK phrases from claim B at evaluation time."""
import re

MOTTE_PHRASES = [
    "i never said", "i didn't say", "i did not say", "what i meant", "i meant was",
    "that's not what i said", "that is not what i said", "to be clear,", "i only said",
    "i only meant", "all i said was", "all i'm saying is", "i wasn't saying", "i was not saying",
    "i'm not saying", "i am not saying",
]
STRAWMAN_PHRASES = [
    "so you're saying", "so you are saying", "so basically you're saying", "what you're saying is",
    "what you are saying is", "are you saying", "you're saying that", "you are saying that",
    "you seem to be saying", "you're essentially saying", "so your argument is", "so what you're saying",
]
PHRASES = {"motte_bailey": MOTTE_PHRASES, "strawman": STRAWMAN_PHRASES}


def mask_triggers(text, shape):
    """Delete the shape's trigger phrases from `text` (crude: the surrounding sentence is left as is)."""
    pat = re.compile("|".join(re.escape(p) for p in sorted(PHRASES[shape], key=len, reverse=True)), re.I)
    return re.sub(r"\s+", " ", pat.sub("", text)).strip()
