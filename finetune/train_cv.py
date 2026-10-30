"""Cross-validated fine-tuning of a DeBERTa-v3-large cross-encoder for the two relational fallacy tasks.

One binary classifier per shape (MOTTE_BAILEY vs NONE, STRAWMAN vs NONE).
Input per example: A [SEP] C [SEP] B (motte-and-bailey) or A [SEP] B (strawman); --input b_only / a_only
gives the single-text ablations.

Evaluation protocol
  * All rows of --data (real, hand-written and synthetic variants) are split by StratifiedGroupKFold grouped
    by conversation_id. Variants carry their source example's conversation_id, so an example and its
    variants always land in the same fold (never one in training and the other in testing).
    --eval_sources restricts the rows used, e.g. reddit_cmv,handwritten for originals only.
  * Optional --aug rows are training-only (kept for compatibility): a row with source_pair_id follows its
    source's fold; a row without one is added to every training fold.
  * --train_frac < 1 keeps a seeded random fraction of the training conversations (learning curve).
  * Each held-out example is also scored with the trigger phrases masked out of claim B (pred_masked).
  * Fixed epoch budget, no early stopping: inner validation splits of ~10 examples are too noisy.

Outputs (results/<tag>/): predictions.csv, summary.csv, run_log.json (GPU, versions, timings, for the compute section).

Usage
  python train_cv.py --data data/dataset_augmented_all_v2.csv --tag nli_pair
  python train_cv.py --data ... --model FacebookAI/roberta-large-mnli --tag roberta_mnli
  python train_cv.py --smoke ...        # tiny run that exercises every code path and measures speed
"""
import argparse
import copy
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from triggers import mask_triggers

HERE = Path(__file__).resolve().parent
NLI_CKPT = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
PLAIN_CKPT = "microsoft/deberta-v3-large"
POSITIVE = {"motte_bailey": "MOTTE_BAILEY", "strawman": "STRAWMAN"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV: pair_id,type_candidate,conversation_id,claim_A,challenge,claim_B,label")
    ap.add_argument("--aug", default="", help="optional training-only CSV(s), comma-separated; same columns (+ optional source_pair_id)")
    ap.add_argument("--shape", default="both", choices=["both", "motte_bailey", "strawman"])
    ap.add_argument("--init", default="nli", choices=["nli", "plain"])
    ap.add_argument("--model", default="", help="any HF sequence-classification checkpoint; overrides --init")
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--input", default="pair", choices=["pair", "b_only", "a_only"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval_sources", default="all",
                    help="'all', or comma-separated `source` values to keep from --data")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_text(row, input_mode, shape, sep, mask_b=False):
    a, c, b = str(row.claim_A), str(row.challenge) if isinstance(row.challenge, str) else "", str(row.claim_B)
    if mask_b:
        b = mask_triggers(b, shape)
    if input_mode == "a_only":
        return a, None
    if input_mode == "b_only":
        return b, None
    if shape == "motte_bailey" and c:
        return a, f"{c} {sep} {b}"
    return a, b


def encode(tok, rows, input_mode, shape, max_len, mask_b=False):
    pairs = [build_text(r, input_mode, shape, tok.sep_token, mask_b) for r in rows.itertuples()]
    first = [p[0] for p in pairs]
    second = [p[1] for p in pairs]
    kwargs = dict(truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return tok(first, **kwargs) if all(s is None for s in second) else tok(first, second, **kwargs)


@torch.no_grad()
def predict(model, tok, rows, input_mode, shape, max_len, device, bs=32, mask_b=False):
    model.eval()
    probs = []
    for i in range(0, len(rows), bs):
        chunk = rows.iloc[i:i + bs]
        enc = encode(tok, chunk, input_mode, shape, max_len, mask_b).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**enc).logits
        probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def train_one(model, tok, train_rows, args, shape, device, seed):
    torch.manual_seed(seed)
    for mod in model.classifier.modules():                          # fresh output layer every run, seeded
        if isinstance(mod, torch.nn.Linear) and mod.out_features == 2:
            torch.nn.init.normal_(mod.weight, std=0.02)
            torch.nn.init.zeros_(mod.bias)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_rows) / args.bs)
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    rng = np.random.default_rng(seed)
    step = 0
    for _ in range(args.epochs):
        order = rng.permutation(len(train_rows))
        for i in range(0, len(order), args.bs):
            batch = train_rows.iloc[order[i:i + args.bs]]
            enc = encode(tok, batch, args.input, shape, args.max_len).to(device)
            y = torch.tensor(batch["y"].values, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if args.smoke and step >= 3:
                return step
    return step


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp, fp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(accuracy=float((y == p).mean()), precision=prec, recall=rec, f1=f1)


def main():
    args = parse_args()
    t_start = time.time()
    device = pick_device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        seeds, args.epochs, args.folds = seeds[:1], 1, 5
    out_dir = HERE / "results" / (args.tag + ("_smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    log = dict(start_utc=datetime.now(timezone.utc).isoformat(), args=vars(args), device=str(device),
               python=platform.python_version(), torch=torch.__version__)
    import transformers
    log["transformers"] = transformers.__version__
    if device.type == "cuda":
        log["gpu"] = torch.cuda.get_device_name(0)
        log["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        log["bf16_supported"] = torch.cuda.is_bf16_supported()
    print(json.dumps({k: v for k, v in log.items() if k != "args"}, indent=1), flush=True)

    data = pd.read_csv(args.data)
    data["challenge"] = data["challenge"].fillna("")
    aug = (pd.concat([pd.read_csv(p) for p in args.aug.split(",")], ignore_index=True)
           if args.aug else pd.DataFrame(columns=data.columns))
    aug["challenge"] = aug["challenge"].fillna("") if len(aug) else aug.get("challenge")
    if "source_pair_id" not in aug.columns:
        aug["source_pair_id"] = ""
    aug["source_pair_id"] = aug["source_pair_id"].fillna("").astype(str)
    if "source" in data.columns and args.eval_sources != "all":
        data = data[data["source"].isin(args.eval_sources.split(","))]
    for col in ("source", "source_pair_id", "kind"):
        data[col] = data[col].fillna("") if col in data.columns else ""
    shapes = ["motte_bailey", "strawman"] if args.shape == "both" else [args.shape]

    t0 = time.time()
    ckpt = args.model or (NLI_CKPT if args.init == "nli" else PLAIN_CKPT)
    tok = AutoTokenizer.from_pretrained(args.model or NLI_CKPT)   # DeBERTa-v3 inits share the NLI ckpt's tokenizer
    # float32 master weights: the NLI checkpoint is stored in fp16, and AdamW on fp16 weights diverges to NaN.
    # bf16 autocast is still used for the forward pass on GPU.
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2, ignore_mismatched_sizes=True,
                                                               dtype=torch.float32)
    assert next(model.parameters()).dtype == torch.float32
    log["checkpoint"] = ckpt
    log["model_load_sec"] = round(time.time() - t0, 1)
    base_model = model.cpu()   # never trained; every fold gets a fresh deep copy (in-place reloading gave NaNs on MPS)
    print(f"model loaded in {log['model_load_sec']}s; {sum(p.numel() for p in model.parameters())/1e6:.0f}M params", flush=True)

    pred_rows, step_times = [], []
    for shape in shapes:
        sub = data[data.type_candidate == shape].reset_index(drop=True).copy()
        sub["y"] = (sub.label == POSITIVE[shape]).astype(int)
        aug_s = aug[aug.type_candidate == shape].copy()
        aug_s["y"] = (aug_s.label == POSITIVE[shape]).astype(int)
        conv_of = dict(zip(sub.pair_id, sub.conversation_id))
        for seed in seeds:
            skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(skf.split(sub, sub.y, groups=sub.conversation_id)):
                if args.smoke and fold >= 1:
                    break
                train_real, test = sub.iloc[tr], sub.iloc[te]
                train_convs = set(train_real.conversation_id)
                follow_mask = aug_s.source_pair_id.map(lambda s: conv_of.get(s) in train_convs).astype(bool)
                follow = aug_s.loc[follow_mask]
                free = aug_s.loc[(aug_s.source_pair_id == "").astype(bool)]
                if args.train_frac < 1:
                    convs = sorted(train_convs)
                    keep = set(np.random.default_rng(seed).choice(convs, max(1, round(args.train_frac * len(convs))), replace=False))
                    train_real = train_real[train_real.conversation_id.isin(keep)]
                    follow = follow[follow.source_pair_id.map(lambda s: conv_of.get(s) in keep).astype(bool)]
                train = pd.concat([train_real, follow, free], ignore_index=True)
                n_aug = len(follow) + len(free)
                test_convs = set(test.conversation_id)
                assert not follow.source_pair_id.map(lambda s: conv_of.get(s) in test_convs).astype(bool).any(), "augmented row leaks into training"
                assert not set(train.pair_id) & set(test.pair_id), "test example in training set"
                if args.smoke:
                    train, test = train.sample(min(16, len(train)), random_state=0), test.head(8)
                model = copy.deepcopy(base_model)
                t1 = time.time()
                steps = train_one(model, tok, train, args, shape, device, seed)
                train_sec = time.time() - t1
                step_times.append(train_sec / max(steps, 1))
                p = predict(model, tok, test, args.input, shape, args.max_len, device)
                pm = predict(model, tok, test, args.input, shape, args.max_len, device, mask_b=True)
                for r, pr, prm in zip(test.itertuples(), p, pm):
                    pred_rows.append(dict(shape=shape, seed=seed, fold=fold, pair_id=r.pair_id, source=r.source,
                                          source_pair_id=r.source_pair_id, kind=r.kind, y=int(r.y), prob=float(pr),
                                          pred=int(pr > 0.5), prob_masked=float(prm), pred_masked=int(prm > 0.5)))
                print(f"[{shape} seed={seed} fold={fold}] train_n={len(train)} (aug {n_aug}) "
                      f"test_n={len(test)} {train_sec:.1f}s", flush=True)
                assert not np.isnan(p).any(), "NaN predictions - stopping instead of writing bad results"
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    preds = pd.DataFrame(pred_rows)
    preds.insert(0, "config", args.tag)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    summ = []
    for (shape, seed), g in preds.groupby(["shape", "seed"]):
        m, mm = metrics(g.y, g.pred), metrics(g.y, g.pred_masked)
        summ.append(dict(config=args.tag, shape=shape, seed=seed, n=len(g), **m,
                         **{f"masked_{k}": v for k, v in mm.items()}))
    summ = pd.DataFrame(summ)
    summ.to_csv(out_dir / "summary.csv", index=False)
    agg = summ.groupby("shape")[["accuracy", "f1", "masked_accuracy", "masked_f1"]].agg(["mean", "std"]).round(3)
    print("\n=== summary (mean/std over seeds) ===\n", agg.to_string(), flush=True)

    log["end_utc"] = datetime.now(timezone.utc).isoformat()
    log["wall_sec"] = round(time.time() - t_start, 1)
    log["sec_per_train_step_mean"] = round(float(np.mean(step_times)), 3) if step_times else None
    if device.type == "cuda":
        log["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (out_dir / "run_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nRESULTS_DIR={out_dir}\nwall {log['wall_sec']}s, {log['sec_per_train_step_mean']} s/step", flush=True)


if __name__ == "__main__":
    main()
