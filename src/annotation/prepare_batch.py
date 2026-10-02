"""
Clean up raw Reddit markup and sample the next labeling batch from the
filtered candidate pools, excluding anything already sampled in a previous
batch (tracked in used_pair_keys.json). Output is a plain CSV you can open
in Excel/Numbers/Google Sheets and fill in the `label` column by hand.
"""

import csv
import html
import json
import random
import re
import sys
from pathlib import Path

IN_DIR = Path("data/processed")
OUT_DIR = Path("data/annotations")
USED_KEYS_PATH = OUT_DIR / "used_pair_keys.json"

N_PER_TYPE = 25


def clean_text(text):
    text = html.unescape(text)
    lines = [ln for ln in text.split("\n") if not ln.strip().startswith(">")]
    text = "\n".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def key_for(r):
    if r["shape"] == "motte_bailey_candidate":
        return f"{r['conversation_id']}|{r['turn_A']}|{r['turn_challenge']}|{r['turn_B']}"
    return f"{r['conversation_id']}|{r['turn_A']}|{r['turn_B']}"


def load_used_keys():
    if USED_KEYS_PATH.exists():
        return set(json.loads(USED_KEYS_PATH.read_text()))
    return set()


def save_used_keys(keys):
    USED_KEYS_PATH.write_text(json.dumps(sorted(keys), indent=2))


def main():
    batch_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    random.seed(42 + batch_num)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    used_keys = load_used_keys()

    motte = load_jsonl(IN_DIR / "candidates_motte_filtered.jsonl")
    straw = load_jsonl(IN_DIR / "candidates_strawman_filtered.jsonl")

    motte_pool = [r for r in motte if key_for(r) not in used_keys]
    straw_pool = [r for r in straw if key_for(r) not in used_keys]

    motte_sample = random.sample(motte_pool, min(N_PER_TYPE, len(motte_pool)))
    straw_sample = random.sample(straw_pool, min(N_PER_TYPE, len(straw_pool)))

    rows = []
    pair_id = (batch_num - 1) * N_PER_TYPE + 1
    for r in motte_sample:
        rows.append({
            "pair_id": f"m{pair_id:03d}",
            "type_candidate": "motte_bailey",
            "conversation_id": r["conversation_id"],
            "speaker_A": r["speaker_A"],
            "claim_A": clean_text(r["text_A"]),
            "challenge": clean_text(r["text_challenge"]),
            "claim_B": clean_text(r["text_B"]),
            "label": "",
            "notes": "",
        })
        used_keys.add(key_for(r))
        pair_id += 1

    pair_id = (batch_num - 1) * N_PER_TYPE + 1
    for r in straw_sample:
        rows.append({
            "pair_id": f"s{pair_id:03d}",
            "type_candidate": "strawman",
            "conversation_id": r["conversation_id"],
            "speaker_A": r["speaker_A"],
            "claim_A": clean_text(r["text_A"]),
            "challenge": "",
            "claim_B": clean_text(r["text_B"]),
            "label": "",
            "notes": "",
        })
        used_keys.add(key_for(r))
        pair_id += 1

    out_path = OUT_DIR / f"batch_{batch_num}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "type_candidate", "conversation_id", "speaker_A",
            "claim_A", "challenge", "claim_B", "label", "notes",
        ])
        writer.writeheader()
        writer.writerows(rows)

    save_used_keys(used_keys)

    print(f"Batch {batch_num}: wrote {len(rows)} rows to {out_path}")
    print(f"  motte-bailey: {len(motte_sample)} (pool remaining before sample: {len(motte_pool)})")
    print(f"  strawman: {len(straw_sample)} (pool remaining before sample: {len(straw_pool)})")


if __name__ == "__main__":
    main()
