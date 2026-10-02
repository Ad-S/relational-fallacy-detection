"""
Compute simple surface-level hedging/specificity features for each claim
pair: does claim_B use more hedging/qualifying language than claim_A (a
narrowing signal), or more absolute language (a broadening signal)?

This is a word-count heuristic, not a model -- deliberately simple, as
described in the proposal (Section 4: "surface hedging cues").

Usage: python3 src/features/specificity_features.py <path/to/batch_N_to_label.csv>
Writes: <same path with _specificity_features.csv suffix>
"""

import csv
import re
import sys
from pathlib import Path

HEDGE_WORDS = {
    "some", "often", "sometimes", "generally", "usually", "may", "might",
    "could", "possibly", "perhaps", "maybe", "a few", "many", "mostly",
    "somewhat", "arguably", "typically", "occasionally", "i think",
    "i believe", "seems", "appears", "tends to", "in some cases",
    "under certain conditions", "not necessarily", "not always",
}

ABSOLUTE_WORDS = {
    "all", "every", "always", "never", "completely", "totally", "only",
    "none", "no one", "everyone", "everything", "nothing", "must",
    "certainly", "definitely", "absolutely", "entirely", "utterly",
}


def count_terms(text: str, terms: set) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(term) for term in terms)


def compute_features(claim_a: str, claim_b: str) -> dict:
    hedge_a = count_terms(claim_a, HEDGE_WORDS)
    hedge_b = count_terms(claim_b, HEDGE_WORDS)
    abs_a = count_terms(claim_a, ABSOLUTE_WORDS)
    abs_b = count_terms(claim_b, ABSOLUTE_WORDS)

    len_a = max(len(claim_a.split()), 1)
    len_b = max(len(claim_b.split()), 1)

    return {
        "hedge_count_A": hedge_a,
        "hedge_count_B": hedge_b,
        "hedge_delta": hedge_b - hedge_a,
        "hedge_rate_delta": round(hedge_b / len_b - hedge_a / len_a, 4),
        "absolute_count_A": abs_a,
        "absolute_count_B": abs_b,
        "absolute_delta": abs_b - abs_a,
        "length_ratio_B_over_A": round(len_b / len_a, 3),
        "explicit_denial_B": int(bool(re.search(
            r"\b(i never said|i didn'?t say|i did not say|i'?m not saying|i am not saying|"
            r"i wasn'?t saying|that'?s not what i said)\b",
            claim_b.lower()
        ))),
    }


def main():
    in_path = Path(sys.argv[1])
    out_path = in_path.parent / (in_path.stem + "_specificity_features.csv")

    with open(in_path) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in rows:
        claim_a = row.get("claim_A", "").strip()
        claim_b = row.get("claim_B", "").strip()
        if not claim_a or not claim_b:
            continue

        feats = compute_features(claim_a, claim_b)
        feats["pair_id"] = row["pair_id"]
        out_rows.append(feats)

    fieldnames = ["pair_id", "hedge_count_A", "hedge_count_B", "hedge_delta",
                  "hedge_rate_delta", "absolute_count_A", "absolute_count_B",
                  "absolute_delta", "length_ratio_B_over_A", "explicit_denial_B"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
