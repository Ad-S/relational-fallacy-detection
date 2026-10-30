"""Merge the 188 real labeled pairs with the 20 hand-written pairs into one dataset (dataset_final_v2.csv).

Hand-written pairs are regular data here (train and test). Each gets its own group id
(conversation_id = handwritten_<pair_id>) so grouped CV can spread them across folds instead of
treating all 20 as one conversation. source = reddit_cmv | handwritten, so results can still be
broken down by source. Inputs are not modified.

Usage: python3 src/annotation/build_dataset_final_v2.py
"""
from pathlib import Path

import pandas as pd

ANNOT = Path("data/annotations")
COLS = ["pair_id", "type_candidate", "conversation_id", "claim_A", "challenge", "claim_B", "label", "source", "source_file"]

real = pd.read_csv(ANNOT / "dataset_consolidated_v2.csv", keep_default_na=False)
hand = pd.read_csv(ANNOT / "synthetic_examples.csv", keep_default_na=False)
hand["conversation_id"] = "handwritten_" + hand["pair_id"]
hand["source"] = "handwritten"
hand["source_file"] = "synthetic_examples.csv"
assert not set(real.pair_id) & set(hand.pair_id)
assert set(hand.label) <= {"MOTTE_BAILEY", "STRAWMAN", "NONE"}

out = pd.concat([real[COLS], hand[COLS]], ignore_index=True)
out.to_csv(ANNOT / "dataset_final_v2.csv", index=False)
print(f"wrote {ANNOT / 'dataset_final_v2.csv'}: {len(out)} rows")
print(pd.crosstab([out.source, out.type_candidate], out.label, margins=True).to_string())
