"""Scripted zero-shot LLM run on all 111 pairs (91 real + 20 synthetic) of dataset_final.csv.

New experiment, NOT the mid-report's baseline. Each example is one fresh, headless Claude Code
session (`claude -p`) with: a replaced system prompt, no tools, no session persistence, no MCP
servers, launched from an empty temporary directory so no CLAUDE.md or project memory is loaded.
Gold labels are dropped before any prompt is built. Raw CLI JSON for every call is saved in raw/.

Usage: venv/bin/python presentation_figures/llm_rerun/run_zeroshot_llm.py [--model M] [--workers N] [--only id,id]
Re-running skips pair_ids that already have a valid raw/<pair_id>.json.
"""
import argparse
import csv
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from prompts import SYSTEM_PROMPT, TEMPLATES, VALID

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "annotations" / "dataset_final.csv"
RAW = HERE / "raw"
OUT = HERE / "zeroshot_rerun_predictions.csv"
INPUT_COLS = ["pair_id", "type_candidate", "source", "claim_A", "challenge", "claim_B"]  # no label


def load_inputs():
    with open(DATA) as f:
        rows = [{k: (r.get(k) or "").strip() for k in INPUT_COLS} for r in csv.DictReader(f)]
    return rows


def parse_label(text, shape):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    label = str(obj.get("label", "")).strip().upper()
    return (label if label in VALID[shape] else None), obj.get("reason")


def call(row, model, cwd):
    prompt = TEMPLATES[row["type_candidate"]].format(**row)
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json",
           "--system-prompt", SYSTEM_PROMPT, "--tools", "", "--no-session-persistence",
           "--strict-mcp-config"]
    for attempt in (1, 2):
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
        rec = {"pair_id": row["pair_id"], "attempt": attempt, "returncode": proc.returncode,
               "stdout": proc.stdout, "stderr": proc.stderr[-2000:], "prompt": prompt, "model_requested": model,
               "time_utc": datetime.now(timezone.utc).isoformat()}
        try:
            out = json.loads(proc.stdout)
            text = out.get("result", "")
        except json.JSONDecodeError:
            out, text = {}, ""
        label, reason = parse_label(text, row["type_candidate"])
        rec.update(parsed_label=label, reason=reason)
        if label:
            break
    (RAW / f"{row['pair_id']}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    rows = load_inputs()
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["pair_id"] in keep]
    todo = []
    for r in rows:
        p = RAW / f"{r['pair_id']}.json"
        if p.exists() and json.loads(p.read_text()).get("parsed_label"):
            continue
        todo.append(r)
    print(f"{len(rows)} selected, {len(todo)} to run, model={args.model}")
    with tempfile.TemporaryDirectory(prefix="zeroshot_cwd_") as cwd:
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(call, r, args.model, cwd): r["pair_id"] for r in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                rec = fu.result()
                print(f"[{i}/{len(todo)}] {rec['pair_id']}: {rec['parsed_label']}", flush=True)

    # assemble predictions from raw files (all selected rows of the full dataset)
    out_rows, failed = [], []
    for r in load_inputs():
        p = RAW / f"{r['pair_id']}.json"
        rec = json.loads(p.read_text()) if p.exists() else {}
        try:
            meta = json.loads(rec.get("stdout") or "{}")
        except json.JSONDecodeError:
            meta = {}
        models_used = ",".join(sorted((meta.get("modelUsage") or {}).keys()))
        if not rec.get("parsed_label"):
            failed.append(r["pair_id"])
        out_rows.append({"pair_id": r["pair_id"], "type_candidate": r["type_candidate"], "source": r["source"],
                         "predicted_label": rec.get("parsed_label") or "", "reason": rec.get("reason") or "",
                         "model_requested": rec.get("model_requested", ""), "models_reported": models_used,
                         "time_utc": rec.get("time_utc", "")})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT.relative_to(ROOT)}; missing/failed: {len(failed)} {failed[:10]}")


if __name__ == "__main__":
    main()
