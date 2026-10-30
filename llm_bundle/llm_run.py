"""Open-weight LLM baselines with vLLM (one model per process): zero-shot classification + stage-1 extraction.

Deterministic: temperature 0, fixed seed, prompts saved alongside the outputs. Zero-shot uses the same
definitions-only prompts as the earlier Claude re-run (zeroshot_prompts.py); extraction prompts are in
extract_prompts.py. Every raw model output is kept in the CSVs.

Usage: python llm_run.py --model Qwen/Qwen2.5-7B-Instruct [--smoke]
Writes: results/llm/<model short name>/{zeroshot.csv, extract.csv, prompts.json, run_log.json}
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

import extract_prompts as EP
import zeroshot_prompts as ZP

HERE = Path(__file__).resolve().parent


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dataset_augmented_all_v2.csv")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    from vllm import LLM, SamplingParams
    import vllm

    df = pd.read_csv(args.data, keep_default_na=False)
    if args.smoke:
        df = df.groupby("type_candidate", group_keys=False).head(5).reset_index(drop=True)
    short = args.model.split("/")[-1] + ("_smoke" if args.smoke else "")
    out_dir = HERE / "results" / "llm" / short
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.90, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=300, seed=0)
    t_load = time.time() - t0

    def chat(system, users):
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": u}] for u in users]
        return [o.outputs[0].text for o in llm.chat(convs, sp, use_tqdm=False)]

    rows = df.to_dict("records")
    zs_prompts = [ZP.TEMPLATES[r["type_candidate"]].format(**r) for r in rows]
    ex_prompts = [EP.TEMPLATES[r["type_candidate"]].format(**r) for r in rows]
    t1 = time.time()
    zs_out = chat(ZP.SYSTEM_PROMPT, zs_prompts)
    ex_out = chat(EP.SYSTEM, ex_prompts)
    t_gen = time.time() - t1

    zs, ex = [], []
    for r, z, e in zip(rows, zs_out, ex_out):
        zo, eo = parse_json(z) or {}, parse_json(e) or {}
        lab = str(zo.get("label", "")).strip().upper()
        zs.append(dict(pair_id=r["pair_id"], type_candidate=r["type_candidate"], source=r.get("source", ""),
                       label=r["label"], predicted_label=lab if lab in ZP.VALID[r["type_candidate"]] else "",
                       reason=zo.get("reason", ""), raw=z))
        if r["type_candidate"] == "strawman":
            ex.append(dict(pair_id=r["pair_id"], type_candidate="strawman", claim_A=r["claim_A"],
                           flag=bool(eo.get("attributes", False)), x_claim=str(eo.get("attributed_claim", "") or ""),
                           defended_claim="", parsed=bool(eo), raw=e))
        else:
            ex.append(dict(pair_id=r["pair_id"], type_candidate="motte_bailey", claim_A=r["claim_A"],
                           flag=bool(eo.get("denies", False)), x_claim=str(eo.get("denied_claim", "") or ""),
                           defended_claim=str(eo.get("defended_claim", "") or ""), parsed=bool(eo), raw=e))
    zs, ex = pd.DataFrame(zs), pd.DataFrame(ex)
    zs.to_csv(out_dir / "zeroshot.csv", index=False)
    ex.to_csv(out_dir / "extract.csv", index=False)
    (out_dir / "prompts.json").write_text(json.dumps({"zeroshot_system": ZP.SYSTEM_PROMPT, "zeroshot": ZP.TEMPLATES,
                                                      "extract_system": EP.SYSTEM, "extract": EP.TEMPLATES}, indent=2))
    log = dict(model=args.model, vllm=vllm.__version__, rows=len(df), load_sec=round(t_load, 1),
               generate_sec=round(t_gen, 1), zeroshot_unparsed=int((zs.predicted_label == "").sum()),
               extract_unparsed=int((~ex.parsed).sum()))
    try:
        import torch
        log["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    (out_dir / "run_log.json").write_text(json.dumps(log, indent=2))
    acc = (zs.predicted_label == zs.label).mean()
    print(json.dumps(log), flush=True)
    print(f"zero-shot accuracy (all rows, raw label match): {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
