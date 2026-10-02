"""
Second pass: narrow the huge candidate pools down using simple keyword phrases
that tend to show up specifically in retreat/clarification (motte-bailey) or
characterization (strawman) moments. Still no LLM/model here — just phrase
matching, to get to a small enough pile a person can actually read.
"""

import json
from pathlib import Path

IN_DIR = Path("data/processed")

# Phrases that tend to appear when someone is narrowing/clarifying what THEY said.
# Checked against text_B (person A's own reply after being challenged).
MOTTE_PHRASES = [
    "i never said",
    "i didn't say",
    "i did not say",
    "what i meant",
    "i meant was",
    "that's not what i said",
    "that is not what i said",
    "to be clear,",
    "i only said",
    "i only meant",
    "all i said was",
    "all i'm saying is",
    "i wasn't saying",
    "i was not saying",
    "i'm not saying",
    "i am not saying",
]

# Phrases that tend to appear when someone is characterizing what the OTHER
# person said. Checked against text_B (the reply/challenger's own words).
STRAWMAN_PHRASES = [
    "so you're saying",
    "so you are saying",
    "so basically you're saying",
    "what you're saying is",
    "what you are saying is",
    "are you saying",
    "you're saying that",
    "you are saying that",
    "you seem to be saying",
    "you're essentially saying",
    "so your argument is",
    "so what you're saying",
]


def matches_any(text, phrases):
    text_lower = text.lower()
    return any(p in text_lower for p in phrases)


def filter_file(in_path, out_path, phrases, text_field):
    n_in, n_out = 0, 0
    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            n_in += 1
            row = json.loads(line)
            if matches_any(row[text_field], phrases):
                f_out.write(json.dumps(row) + "\n")
                n_out += 1
    return n_in, n_out


def main():
    n_in, n_out = filter_file(
        IN_DIR / "candidates_motte.jsonl",
        IN_DIR / "candidates_motte_filtered.jsonl",
        MOTTE_PHRASES,
        text_field="text_B",
    )
    print(f"Motte-bailey candidates: {n_in} -> {n_out} after phrase filter")

    n_in, n_out = filter_file(
        IN_DIR / "candidates_strawman.jsonl",
        IN_DIR / "candidates_strawman_filtered.jsonl",
        STRAWMAN_PHRASES,
        text_field="text_B",
    )
    print(f"Strawman candidates: {n_in} -> {n_out} after phrase filter")


if __name__ == "__main__":
    main()
