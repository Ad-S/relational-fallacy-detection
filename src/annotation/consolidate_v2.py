"""Consolidate ALL hand-labeled batches into dataset_consolidated_v2.csv.

Successor to consolidate_dataset.py (left untouched so the mid-report dataset stays reproducible).
Differences: it accepts the label spellings used in later batches, keys everything on pair_id (file
names do not matter), refuses anything it does not recognise, and writes a row-by-row audit file.

Label normalisation (case-insensitive, whitespace-stripped):
  motte_bailey                          -> MOTTE_BAILEY   (only on motte-and-bailey-shaped rows)
  strawman                              -> STRAWMAN       (only on strawman-shaped rows)
  none | not_motte_bailey | not_strawman-> NONE           (not_* only on the matching shape)
  uncertain | insufficient_context      -> dropped (UNCERTAIN)
  (blank)                               -> dropped (unlabeled)
  anything else, or a label on the wrong shape -> error, nothing is written.
Rows whose extraction_status is not OK (blank claims) are dropped whatever their label says.
Claim text is whitespace-collapsed; it is verified to equal the original batch_N_to_label.csv text
once whitespace is ignored (so no content was changed while labeling). The one tolerated difference is
a leading apostrophe that spreadsheets strip ('Natural' -> Natural'); the text the annotator saw is kept
and those rows are marked in the audit file (note column).

Usage: python3 src/annotation/consolidate_v2.py
Writes: data/annotations/dataset_consolidated_v2.csv, dataset_consolidated_v2_label_audit.csv
"""
import glob
import re
import sys
from pathlib import Path

import pandas as pd

ANNOT = Path("data/annotations")
OUT = ANNOT / "dataset_consolidated_v2.csv"
AUDIT = ANNOT / "dataset_consolidated_v2_label_audit.csv"
POSITIVE = {"motte_bailey": "MOTTE_BAILEY", "strawman": "STRAWMAN"}
NEG_SPELLING = {"motte_bailey": "not_motte_bailey", "strawman": "not_strawman"}
DROP = {"uncertain", "insufficient_context", ""}


def ws(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def map_label(raw, shape):
    k = raw.strip().lower()
    if k in DROP:
        return None, "uncertain_or_unlabeled" if k else "unlabeled"
    if k == "none" or k == NEG_SPELLING[shape]:
        return "NONE", ""
    if k == POSITIVE[shape].lower():
        return POSITIVE[shape], ""
    raise ValueError(f"label {raw!r} is not valid for a {shape}-shaped row")


def main():
    orig = {}
    for p in sorted(glob.glob(str(ANNOT / "batch_[0-9]*_to_label.csv"))):
        for r in pd.read_csv(p, keep_default_na=False).itertuples():
            orig[r.pair_id] = (Path(p).name, ws(r.claim_A), ws(r.challenge), ws(r.claim_B))

    kept, audit, errors, seen = [], [], [], {}
    for p in sorted(glob.glob(str(ANNOT / "annotated_batch_*_to_label.csv"))):
        name = Path(p).name
        for r in pd.read_csv(p, keep_default_na=False).itertuples():
            if r.pair_id in seen:
                errors.append(f"{r.pair_id} appears in both {seen[r.pair_id]} and {name}")
                continue
            seen[r.pair_id] = name
            if r.pair_id not in orig:
                errors.append(f"{r.pair_id} ({name}) matches no batch_N_to_label.csv")
                continue
            row = dict(pair_id=r.pair_id, file=name, original_batch_file=orig[r.pair_id][0], shape=r.type_candidate,
                       extraction_status=r.extraction_status, raw_label=r.label, mapped_label="", dropped_reason="", note="")
            try:
                label, why = map_label(r.label, r.type_candidate)
            except ValueError as e:
                errors.append(f"{r.pair_id} ({name}): {e}")
                audit.append(row)
                continue
            claims = (ws(r.claim_A), ws(r.challenge), ws(r.claim_B))
            if tuple(c.lstrip("'") for c in claims) != tuple(c.lstrip("'") for c in orig[r.pair_id][1:]):
                errors.append(f"{r.pair_id} ({name}): claim text differs from {orig[r.pair_id][0]} beyond whitespace")
            if claims != orig[r.pair_id][1:]:
                row["note"] = "leading apostrophe stripped by spreadsheet"
            if r.extraction_status != "OK":
                row["dropped_reason"] = f"extraction_status={r.extraction_status}"
            elif not claims[0] or not claims[2] or (r.type_candidate == "motte_bailey" and not claims[1]):
                row["dropped_reason"] = "missing claim text"
            elif label is None:
                row["dropped_reason"] = why
            else:
                row["mapped_label"] = label
                kept.append(dict(pair_id=r.pair_id, type_candidate=r.type_candidate, conversation_id=r.conversation_id,
                                 claim_A=claims[0], challenge=claims[1], claim_B=claims[2], label=label,
                                 source="reddit_cmv", source_file=name))
            audit.append(row)

    if errors:
        print("ERRORS - nothing written:")
        for e in errors:
            print("  ", e)
        sys.exit(1)

    df = pd.DataFrame(kept).sort_values("pair_id").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    pd.DataFrame(audit).to_csv(AUDIT, index=False)
    print(f"wrote {OUT} ({len(df)} rows) and {AUDIT} ({len(audit)} rows)")
    print(pd.crosstab(df.type_candidate, df.label, margins=True).to_string())


if __name__ == "__main__":
    main()
