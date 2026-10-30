"""Turn a reviewed variant sheet into data.

  --set handwritten (default): data/augmented/handwritten_variants_for_review.csv
        -> handwritten_variants_accepted.csv and data/annotations/dataset_augmented_v2.csv
           (dataset_final_v2 + hand-written variants)
  --set real: data/augmented/real_variants_for_review.csv
        -> real_variants_accepted.csv and data/annotations/dataset_augmented_all_v2.csv
           (dataset_final_v2 + hand-written variants (if collected) + real variants)

A variant is kept only if `accept` is y/yes/1/true; blank means not reviewed and is left out.
`corrected_label` (optional) overrides the intended label.
  --accept-all  keep every variant with its generated label (no human review); label_verified=False.
Each kept variant inherits its source example's group (conversation_id), so grouped CV keeps a variant
in the same fold as the example it came from. Accepted files carry source_pair_id (usable as --aug for
finetune/train_cv.py).

Usage: python3 src/augmentation/collect_accepted.py [--set handwritten|real] [--accept-all]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUG, ANNOT = ROOT / "data" / "augmented", ROOT / "data" / "annotations"
BASE = ANNOT / "dataset_final_v2.csv"
SETS = {"handwritten": ("handwritten_variant", AUG / "handwritten_variants_for_review.csv",
                        AUG / "handwritten_variants_accepted.csv", ANNOT / "dataset_augmented_v2.csv"),
        "real": ("real_variant", AUG / "real_variants_for_review.csv",
                 AUG / "real_variants_accepted.csv", ANNOT / "dataset_augmented_all_v2.csv")}
VALID = {"motte_bailey": {"MOTTE_BAILEY", "NONE"}, "strawman": {"STRAWMAN", "NONE"}}
YES, NO = {"y", "yes", "1", "true"}, {"n", "no", "0", "false"}


def collect(review, source_value, accept_all, base):
    rev = pd.read_csv(review, keep_default_na=False)
    if accept_all:
        rev["accept"], rev["corrected_label"] = "y", ""
    decision = rev.accept.str.strip().str.lower()
    bad = rev[~decision.isin(YES | NO | {""})]
    if len(bad):
        sys.exit(f"unrecognised accept values: {bad[['variant_id', 'accept']].to_dict('records')}")
    keep = rev[decision.isin(YES)].copy()
    keep["label"] = keep.corrected_label.str.strip().str.upper().where(keep.corrected_label.str.strip() != "",
                                                                        keep.label_intended)
    wrong = keep[[lab not in VALID[s] for lab, s in zip(keep.label, keep.type_candidate)]]
    if len(wrong):
        sys.exit(f"labels invalid for their shape: {wrong[['variant_id', 'type_candidate', 'label']].to_dict('records')}")
    conv = dict(zip(base.pair_id, base.conversation_id))
    out = pd.DataFrame({"pair_id": keep.variant_id, "type_candidate": keep.type_candidate,
                        "conversation_id": keep.source_pair_id.map(conv), "claim_A": keep.claim_A,
                        "challenge": keep.challenge, "claim_B": keep.claim_B, "label": keep.label,
                        "source": source_value, "source_file": review.name, "source_pair_id": keep.source_pair_id,
                        "kind": keep.kind, "label_verified": not accept_all})
    assert out.conversation_id.notna().all(), "variant whose source is not in dataset_final_v2.csv"
    print("accept-all: every variant kept with its generated label (not human-reviewed)" if accept_all else
          f"reviewed {(decision != '').sum()}/{len(rev)}: accepted {len(keep)}, rejected {decision.isin(NO).sum()}, "
          f"not reviewed {(decision == '').sum()}; label corrected on {(keep.corrected_label.str.strip() != '').sum()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="handwritten", choices=list(SETS))
    ap.add_argument("--accept-all", action="store_true")
    args = ap.parse_args()
    source_value, review, accepted_path, merged_path = SETS[args.set]
    base = pd.read_csv(BASE, keep_default_na=False)
    out = collect(review, source_value, args.accept_all, base)
    out.to_csv(accepted_path, index=False)
    parts = [base.assign(source_pair_id="", kind="original", label_verified=True)]
    if args.set == "real" and SETS["handwritten"][2].exists():
        parts.append(pd.read_csv(SETS["handwritten"][2], keep_default_na=False))
    parts.append(out)
    merged = pd.concat(parts, ignore_index=True)
    assert not merged.pair_id.duplicated().any()
    merged.to_csv(merged_path, index=False)
    print(f"wrote {accepted_path.name} ({len(out)}) and {merged_path.name} ({len(merged)} rows)")
    print(pd.crosstab([merged.source, merged.type_candidate], merged.label, margins=True).to_string())


if __name__ == "__main__":
    main()
