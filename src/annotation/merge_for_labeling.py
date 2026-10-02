"""
Merge a batch's raw candidate CSV (conversation_id, speaker) with its
extracted-claims CSV (clean claims + extraction status/notes) into a single
file that's actually ready to hand-label: one row per candidate, with the
clean claim text AND an empty `label` + `annotator_notes` column.
"""

import csv
import sys
from pathlib import Path

OUT_DIR = Path("data/annotations")


def main():
    batch_num = int(sys.argv[1])
    raw_path = OUT_DIR / f"batch_{batch_num}.csv"
    extracted_path = OUT_DIR / f"batch_{batch_num}_extracted.csv"
    out_path = OUT_DIR / f"batch_{batch_num}_to_label.csv"

    with open(raw_path) as f:
        raw_rows = {r["pair_id"]: r for r in csv.DictReader(f)}
    with open(extracted_path) as f:
        extracted_rows = {r["pair_id"]: r for r in csv.DictReader(f)}

    fieldnames = [
        "pair_id", "type_candidate", "conversation_id", "speaker_A",
        "extraction_status", "claim_A", "challenge", "claim_B",
        "label", "annotator_notes", "extraction_notes",
    ]

    out_rows = []
    for pair_id, raw in raw_rows.items():
        ext = extracted_rows.get(pair_id, {})
        out_rows.append({
            "pair_id": pair_id,
            "type_candidate": raw["type_candidate"],
            "conversation_id": raw["conversation_id"],
            "speaker_A": raw["speaker_A"],
            "extraction_status": ext.get("status", "MISSING"),
            "claim_A": ext.get("claim_A_extracted") or "",
            "challenge": ext.get("challenge_extracted") or "",
            "claim_B": ext.get("claim_B_extracted") or "",
            "label": "",
            "annotator_notes": "",
            "extraction_notes": ext.get("notes", ""),
        })

    # keep pair_id sort order stable (m### then s###)
    out_rows.sort(key=lambda r: r["pair_id"])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    labelable = sum(1 for r in out_rows if r["extraction_status"] == "OK")
    print(f"Wrote {len(out_rows)} rows to {out_path}")
    print(f"  labelable (status=OK): {labelable}")
    print(f"  skip/low-confidence: {len(out_rows) - labelable}")


if __name__ == "__main__":
    main()
