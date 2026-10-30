"""NLI features on GPU - same definition as src/features/nli_features.py, batched.

For every pair: entailment/neutral/contradiction probabilities for A -> B (premise A, hypothesis B) and
B -> A, from the frozen MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (max_length 256,
rounded to 4 decimals). Loaded in float32. If --check is given, rows that also appear in that CSV are
compared and the max absolute difference is reported (small differences come from batching/precision).

Usage: python features_gpu.py --data data/dataset_augmented_all_v2.csv [--check data/<old>_nli_features.csv]
Writes: results/features/<data stem>_nli_features.csv
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CKPT = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


@torch.no_grad()
def score(model, tok, premises, hypotheses, device, bs=32):
    idx = {v.lower(): k for k, v in model.config.id2label.items()}
    rows = []
    for i in range(0, len(premises), bs):
        enc = tok(premises[i:i + bs], hypotheses[i:i + bs], truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits.float(), -1).cpu()
        for p in probs:
            rows.append((p[idx["entailment"]].item(), p[idx["neutral"]].item(), p[idx["contradiction"]].item()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--check", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data, keep_default_na=False)
    df = df[(df.claim_A.str.strip() != "") & (df.claim_B.str.strip() != "")]
    if args.smoke:
        df = df.head(16)
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT, dtype=torch.float32).to(device).eval()
    a, b = df.claim_A.str.strip().tolist(), df.claim_B.str.strip().tolist()
    ab, ba = score(model, tok, a, b, device), score(model, tok, b, a, device)
    out = pd.DataFrame({"pair_id": df.pair_id.values,
                        "entail_AB": [round(x[0], 4) for x in ab], "neutral_AB": [round(x[1], 4) for x in ab],
                        "contra_AB": [round(x[2], 4) for x in ab], "entail_BA": [round(x[0], 4) for x in ba],
                        "neutral_BA": [round(x[1], 4) for x in ba], "contra_BA": [round(x[2], 4) for x in ba]})
    out_dir = HERE / "results" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(args.data).stem}{'_smoke' if args.smoke else ''}_nli_features.csv"
    out.to_csv(path, index=False)
    log = {"rows": len(out), "sec": round(time.time() - t0, 1), "device": str(device)}
    if args.check:
        old = pd.read_csv(args.check)
        m = out.merge(old, on="pair_id", suffixes=("", "_old"))
        cols = ["entail_AB", "neutral_AB", "contra_AB", "entail_BA", "neutral_BA", "contra_BA"]
        log["check_rows"] = len(m)
        log["check_max_abs_diff"] = float(max((m[c] - m[c + "_old"]).abs().max() for c in cols)) if len(m) else None
    print(json.dumps(log), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
