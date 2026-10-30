"""Layer-wise linear probing of frozen encoders: where is the fallacy information?

For each checkpoint and each shape, the input is built exactly as in train_cv.py (A [SEP] C [SEP] B, or
A [SEP] B). Hidden states of every layer are mean-pooled over tokens; a standardised logistic regression
is trained per layer with StratifiedGroupKFold (5 folds, grouped by conversation_id, same as fine-tuning)
for several seeds. Reports accuracy, F1 and ROC-AUC per layer (mean/std over seeds).

Usage: python probe_layers.py --data data/dataset_augmented_all_v2.csv
Writes: results/probing/probe_results.csv, run_log.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from train_cv import NLI_CKPT, PLAIN_CKPT, POSITIVE, encode

HERE = Path(__file__).resolve().parent
MODELS = {"deberta_nli": (NLI_CKPT, NLI_CKPT), "deberta_plain": (PLAIN_CKPT, NLI_CKPT)}   # (weights, tokenizer)


@torch.no_grad()
def layer_features(model, tok, rows, shape, device, bs=16):
    feats = None
    for i in range(0, len(rows), bs):
        enc = encode(tok, rows.iloc[i:i + bs], "pair", shape, 256).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states          # (layers+1) x [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = torch.stack([(h.float() * mask).sum(1) / mask.sum(1) for h in hs], 0).cpu().numpy()  # L x B x H
        feats = pooled if feats is None else np.concatenate([feats, pooled], 1)
    return feats


def probe(X, y, groups, seeds):
    out = []
    for seed in seeds:
        pred, prob = np.zeros(len(y), int), np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            clf.fit(X[tr], y[tr])
            prob[te] = clf.predict_proba(X[te])[:, 1]
            pred[te] = (prob[te] > 0.5).astype(int)
        out.append(dict(accuracy=float((pred == y).mean()), f1=float(f1_score(y, pred, zero_division=0)),
                        auc=float(roc_auc_score(y, prob))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default="deberta_nli,deberta_plain")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    data = pd.read_csv(args.data, keep_default_na=False)
    rows = []
    for name in args.models.split(","):
        weights, tok_name = MODELS[name]
        tok = AutoTokenizer.from_pretrained(tok_name)
        model = AutoModel.from_pretrained(weights, dtype=torch.float32).to(device).eval()
        for shape, pos in POSITIVE.items():
            sub = data[data.type_candidate == shape].reset_index(drop=True)
            if args.smoke:
                sub = sub.groupby("label", group_keys=False).head(15).reset_index(drop=True)
            y = (sub.label == pos).astype(int).values
            feats = layer_features(model, tok, sub, shape, device)
            for layer in range(feats.shape[0]):
                res = probe(feats[layer], y, sub.conversation_id.values, seeds[:1] if args.smoke else seeds)
                for k in ("accuracy", "f1", "auc"):
                    vals = [r[k] for r in res]
                    rows.append(dict(model=name, shape=shape, layer=layer, n=len(y), metric=k,
                                     mean=float(np.mean(vals)), std=float(np.std(vals))))
            best = max((r for r in rows if r["model"] == name and r["shape"] == shape and r["metric"] == "auc"),
                       key=lambda r: r["mean"])
            print(f"[{name} {shape}] n={len(y)} layers={feats.shape[0]} best AUC {best['mean']:.3f} at layer {best['layer']}",
                  flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out_dir = HERE / "results" / ("probing_smoke" if args.smoke else "probing")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "probe_results.csv", index=False)
    (out_dir / "run_log.json").write_text(json.dumps({"sec": round(time.time() - t0, 1), "device": str(device),
                                                      "args": vars(args)}, indent=2))
    print(f"wrote {out_dir / 'probe_results.csv'} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
