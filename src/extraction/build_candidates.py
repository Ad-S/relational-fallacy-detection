"""
Pull candidate turn-pairs out of the CMV corpus using pure reply-tree shape,
no NLP/LLM involved yet. Two shapes:

  motte-bailey candidate:  A --(challenge, different speaker)--> C --(same speaker as A)--> B
  strawman candidate:      A --(direct reply, different speaker)--> B

Output: two JSONL files in data/processed/, one row per candidate pair.
"""

import json
from pathlib import Path

from convokit import Corpus

CORPUS_PATH = "data/raw/winning-args-corpus"
OUT_DIR = Path("data/processed")
MIN_CHARS = 40
BAD_TEXT = {"[deleted]", "[removed]"}


def is_usable(utt):
    text = (utt.text or "").strip()
    if text in BAD_TEXT:
        return False
    if len(text) < MIN_CHARS:
        return False
    return True


def build_children_map(convo):
    children = {}
    for utt in convo.iter_utterances():
        if utt.reply_to is not None:
            children.setdefault(utt.reply_to, []).append(utt.id)
    return children


def main():
    corpus = Corpus(filename=CORPUS_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    motte_out = open(OUT_DIR / "candidates_motte.jsonl", "w")
    straw_out = open(OUT_DIR / "candidates_strawman.jsonl", "w")

    n_motte, n_straw, n_convos = 0, 0, 0

    for convo in corpus.iter_conversations():
        n_convos += 1
        utt_by_id = {u.id: u for u in convo.iter_utterances()}
        children = build_children_map(convo)

        for a_id, a in utt_by_id.items():
            if not is_usable(a):
                continue

            for c_id in children.get(a_id, []):
                c = utt_by_id[c_id]

                # --- strawman candidate: A -> B (direct reply, different speaker) ---
                if c.speaker.id != a.speaker.id and is_usable(c):
                    row = {
                        "conversation_id": convo.id,
                        "shape": "strawman_candidate",
                        "turn_A": a_id,
                        "turn_B": c_id,
                        "speaker_A": a.speaker.id,
                        "speaker_B": c.speaker.id,
                        "text_A": a.text,
                        "text_B": c.text,
                        "turn_distance": 1,
                    }
                    straw_out.write(json.dumps(row) + "\n")
                    n_straw += 1

                # --- motte-bailey candidate: A -> C (challenge) -> B (same speaker as A) ---
                if c.speaker.id != a.speaker.id:
                    for b_id in children.get(c_id, []):
                        b = utt_by_id[b_id]
                        if b.speaker.id == a.speaker.id and is_usable(b):
                            row = {
                                "conversation_id": convo.id,
                                "shape": "motte_bailey_candidate",
                                "turn_A": a_id,
                                "turn_challenge": c_id,
                                "turn_B": b_id,
                                "speaker_A": a.speaker.id,
                                "speaker_challenge": c.speaker.id,
                                "text_A": a.text,
                                "text_challenge": c.text,
                                "text_B": b.text,
                                "turn_distance": 2,
                            }
                            motte_out.write(json.dumps(row) + "\n")
                            n_motte += 1

    motte_out.close()
    straw_out.close()

    print(f"Conversations scanned: {n_convos}")
    print(f"Motte-and-bailey candidates: {n_motte}")
    print(f"Strawman candidates: {n_straw}")


if __name__ == "__main__":
    main()
