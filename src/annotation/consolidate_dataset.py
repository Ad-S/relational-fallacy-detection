"""
Merge all annotated_batch_N_to_label.csv files into one clean dataset,
ready for feature extraction and modeling.

- Drops rows with no label, or label UNCERTAIN (not used for training/eval).
- Normalizes label text (case, stray variants like "maybe none").
- Adds a `source` column (reddit_cmv for now; synthetic_handwritten rows
  can be appended later via append_synthetic.py).
- Fails loudly on any label it doesn't recognize, rather than silently
  dropping or guessing.
"""

import csv
import glob
import sys
from pathlib import Path

ANNOT_DIR = Path("data/annotations")
OUT_PATH = ANNOT_DIR / "dataset_consolidated.csv"

VALID_LABELS = {"MOTTE_BAILEY", "STRAWMAN", "NONE"}
DROP_LABELS = {"UNCERTAIN", ""}


def normalize_label(raw: str) -> str:
    cleaned = raw.strip().upper()
    if cleaned in VALID_LABELS or cleaned in DROP_LABELS:
        return cleaned
    # known typo variants -> ask rather than guess silently
    raise ValueError(f"Unrecognized label: {raw!r}")


def main():
    pattern = str(ANNOT_DIR / "annotated_batch_*_to_label.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched {pattern}")
        sys.exit(1)

    kept, dropped_uncertain, dropped_blank = [], 0, 0
    unrecognized = []

    for path in files:
        with open(path) as f:
            for row in csv.DictReader(f):
                raw_label = row.get("label", "")
                try:
                    label = normalize_label(raw_label)
                except ValueError:
                    unrecognized.append((path, row["pair_id"], raw_label))
                    continue

                if label == "UNCERTAIN":
                    dropped_uncertain += 1
                    continue
                if label == "":
                    dropped_blank += 1
                    continue

                kept.append({
                    "pair_id": row["pair_id"],
                    "type_candidate": row["type_candidate"],
                    "conversation_id": row["conversation_id"],
                    "claim_A": row["claim_A"],
                    "challenge": row.get("challenge", ""),
                    "claim_B": row["claim_B"],
                    "label": label,
                    "source": "reddit_cmv",
                    "source_file": Path(path).name,
                })

    if unrecognized:
        print("UNRECOGNIZED LABELS FOUND -- fix these before continuing:")
        for path, pair_id, raw in unrecognized:
            print(f"  {path}  pair_id={pair_id}  label={raw!r}")
        sys.exit(1)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "type_candidate", "conversation_id",
            "claim_A", "challenge", "claim_B", "label", "source", "source_file",
        ])
        writer.writeheader()
        writer.writerows(kept)

    from collections import Counter
    label_counts = Counter(r["label"] for r in kept)
    type_counts = Counter(r["type_candidate"] for r in kept)

    print(f"Consolidated {len(files)} files -> {OUT_PATH}")
    print(f"Kept: {len(kept)}  |  dropped UNCERTAIN: {dropped_uncertain}  |  dropped blank: {dropped_blank}")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Type counts: {dict(type_counts)}")


if __name__ == "__main__":
    main()
