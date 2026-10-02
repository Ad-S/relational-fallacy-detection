"""
Compute NLI (entailment/neutral/contradiction) scores for every labeled claim
pair, in both directions (A implies B? B implies A?). Uses a frozen,
pretrained NLI model -- nothing here is trained on our data, it's just a
feature extractor, exactly as described in the proposal (Section 4, citing
Laurer et al. 2024).

Usage: python3 src/features/nli_features.py <path/to/batch_N_to_label.csv>
Writes: <same path with _nli_features.csv suffix>
"""

import csv
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

print(f"Loading {MODEL_NAME} (first run downloads ~1.7GB, cached after)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

id2label = model.config.id2label  # e.g. {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
label2idx = {v.lower(): k for k, v in id2label.items()}


def score(premise: str, hypothesis: str) -> dict:
    """Returns {'entailment': p, 'neutral': p, 'contradiction': p} for premise -> hypothesis."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {
        "entailment": probs[label2idx["entailment"]].item(),
        "neutral": probs[label2idx["neutral"]].item(),
        "contradiction": probs[label2idx["contradiction"]].item(),
    }


def main():
    in_path = Path(sys.argv[1])
    out_path = in_path.parent / (in_path.stem + "_nli_features.csv")

    with open(in_path) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for i, row in enumerate(rows):
        claim_a = row.get("claim_A", "").strip()
        claim_b = row.get("claim_B", "").strip()
        if not claim_a or not claim_b:
            continue  # skip SPRAWLING_SKIP/LOW_CONFIDENCE rows with blank claims

        ab = score(claim_a, claim_b)   # does A imply B?
        ba = score(claim_b, claim_a)   # does B imply A?

        out_rows.append({
            "pair_id": row["pair_id"],
            "entail_AB": round(ab["entailment"], 4),
            "neutral_AB": round(ab["neutral"], 4),
            "contra_AB": round(ab["contradiction"], 4),
            "entail_BA": round(ba["entailment"], 4),
            "neutral_BA": round(ba["neutral"], 4),
            "contra_BA": round(ba["contradiction"], 4),
        })
        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{len(rows)} rows...")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "entail_AB", "neutral_AB", "contra_AB",
            "entail_BA", "neutral_BA", "contra_BA",
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
