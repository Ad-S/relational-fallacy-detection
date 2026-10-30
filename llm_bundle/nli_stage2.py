"""Stage 2 of the two-stage pipeline: does claim A entail the proposition extracted in stage 1?

For every results/llm/<model>/extract.csv: P(entailment) for A -> x_claim (attributed claim for strawman rows,
denied claim for motte-and-bailey rows), plus A -> defended_claim and defended_claim -> A for
motte-and-bailey rows. Frozen MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli in float32.
Empty propositions get NaN. Decision rules are applied later in the analysis, not here.

Usage: python nli_stage2.py [--smoke]
Writes: results/llm/<model>/stage2.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CKPT = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


@torch.no_grad()
def entail(model, tok, prem, hyp, device, bs=32):
    idx = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]
    out = np.full(len(prem), np.nan)
    keep = [i for i, (p, h) in enumerate(zip(prem, hyp)) if str(p).strip() and str(h).strip()]
    for j in range(0, len(keep), bs):
        ids = keep[j:j + bs]
        enc = tok([prem[i] for i in ids], [hyp[i] for i in ids], truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to(device)
        out[ids] = torch.softmax(model(**enc).logits.float(), -1)[:, idx].cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT, dtype=torch.float32).to(device).eval()
    files = sorted((HERE / "results" / "llm").glob("*/extract.csv"))
    files = [f for f in files if f.parent.name.endswith("_smoke") == args.smoke]
    if not files:
        raise SystemExit("no extract.csv files found")
    for f in files:
        ex = pd.read_csv(f, keep_default_na=False)
        a = ex.claim_A.tolist()
        ex["entail_A_to_x"] = entail(model, tok, a, ex.x_claim.tolist(), device)
        ex["entail_A_to_defended"] = entail(model, tok, a, ex.defended_claim.tolist(), device)
        ex["entail_defended_to_A"] = entail(model, tok, ex.defended_claim.tolist(), a, device)
        ex.drop(columns=["raw"]).to_csv(f.parent / "stage2.csv", index=False)
        print(f"{f.parent.name}: {len(ex)} rows, flagged {int(ex.flag.astype(str).str.lower().eq('true').sum())}, "
              f"x_claim present {int((ex.x_claim.str.strip() != '').sum())}", flush=True)


if __name__ == "__main__":
    main()
