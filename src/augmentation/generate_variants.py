"""Generate variants of labeled examples and write a review sheet (nothing is added to any dataset here).
  --seeds handwritten (default): 5 per example - 2 paraphrase, 2 topic_transfer, 1 label flip.
  --seeds real: 2 per real CMV example - 1 paraphrase, 1 topic_transfer; no flips, so every label stays
                the human label. Raw outputs in data/augmented/raw_real/, sheet real_variants_for_review.csv.

Each seed is one fresh headless Claude Code session (`claude -p`): replaced system prompt, no tools, no
session persistence, no MCP servers, launched from an empty temp dir. Raw outputs: data/augmented/raw/.
Default generator: claude-sonnet-5-5 (different from the Opus 5.5 zero-shot baseline).

Usage: venv/bin/python src/augmentation/generate_variants.py [--seeds handwritten|real] [--model M] [--only id,id]
Writes: data/augmented/handwritten_variants_for_review.csv or real_variants_for_review.csv
Re-running skips seeds that already have a valid raw file.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import SYSTEM_PROMPT, build_prompt, build_prompt_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "data" / "annotations" / "dataset_final_v2.csv"
AUG = ROOT / "data" / "augmented"
SETS = {  # seed source -> (source value in dataset_final_v2, raw dir, review sheet, expected kinds, prompt builder)
    "handwritten": ("handwritten", AUG / "raw", AUG / "handwritten_variants_for_review.csv",
                    ["paraphrase", "paraphrase", "topic_transfer", "topic_transfer", "flip"], build_prompt),
    "real": ("reddit_cmv", AUG / "raw_real", AUG / "real_variants_for_review.csv",
             ["paraphrase", "topic_transfer"], build_prompt_real),
}
SRC, RAW, OUT, KINDS, BUILD = SETS["handwritten"]


def ws(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def load_seeds():
    with open(SEEDS) as f:
        return [r for r in csv.DictReader(f) if r["source"] == SRC]


def validate(variants, seed, flip_label):
    issues = []
    if len(variants) != len(KINDS):
        issues.append(f"expected {len(KINDS)} variants, got {len(variants)}")
    kinds = [v.get("kind") for v in variants]
    if sorted(kinds) != sorted(KINDS):
        issues.append(f"kinds {kinds}")
    per = []
    for v in variants:
        p = []
        want = flip_label if v.get("kind") == "flip" else seed["label"]
        if v.get("label") != want:
            p.append(f"label {v.get('label')} != {want}")
        if not ws(v.get("claim_A")) or not ws(v.get("claim_B")):
            p.append("empty claim")
        if seed["type_candidate"] == "motte_bailey" and not ws(v.get("challenge")):
            p.append("missing challenge")
        if seed["type_candidate"] == "strawman" and ws(v.get("challenge")):
            p.append("strawman with challenge")
        if v.get("kind") == "flip":
            if ws(v.get("claim_A")) != ws(seed["claim_A"]):
                p.append("flip changed claim A")
            if ws(v.get("challenge")) != ws(seed["challenge"]):
                p.append("flip changed challenge")
        per.append("; ".join(p) or "ok")
    return issues, per


def run_seed(seed, model, cwd):
    prompt, flip_label = BUILD(seed["type_candidate"], ws(seed["claim_A"]), ws(seed["challenge"]),
                                      ws(seed["claim_B"]), seed["label"])
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--system-prompt", SYSTEM_PROMPT,
           "--tools", "", "--no-session-persistence", "--strict-mcp-config"]
    rec = {}
    for attempt in (1, 2):
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
        rec = {"seed": seed["pair_id"], "attempt": attempt, "model_requested": model, "prompt": prompt,
               "stdout": proc.stdout, "stderr": proc.stderr[-2000:], "time_utc": datetime.now(timezone.utc).isoformat(),
               "flip_label": flip_label, "variants": None}
        try:
            text = json.loads(proc.stdout).get("result", "")
            obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            rec["variants"] = obj["variants"]
            break
        except Exception as e:  # noqa: BLE001 - record and retry once
            rec["parse_error"] = repr(e)
    (RAW / f"{seed['pair_id']}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="handwritten", choices=list(SETS))
    args = ap.parse_args()
    global SRC, RAW, OUT, KINDS, BUILD
    SRC, RAW, OUT, KINDS, BUILD = SETS[args.seeds]
    RAW.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds()
    if args.only:
        seeds = [s for s in seeds if s["pair_id"] in set(args.only.split(","))]
    todo = [s for s in seeds if not ((RAW / f"{s['pair_id']}.json").exists()
                                     and json.loads((RAW / f"{s['pair_id']}.json").read_text()).get("variants"))]
    print(f"{len(seeds)} seeds, {len(todo)} to generate with {args.model}")
    with tempfile.TemporaryDirectory(prefix="augment_cwd_") as cwd, ThreadPoolExecutor(args.workers) as ex:
        for fu in as_completed([ex.submit(run_seed, s, args.model, cwd) for s in todo]):
            r = fu.result()
            print(f"  {r['seed']}: {'ok' if r['variants'] else 'FAILED ' + r.get('parse_error', '')}", flush=True)

    rows, failed = [], []
    for s in seeds:
        p = RAW / f"{s['pair_id']}.json"
        rec = json.loads(p.read_text()) if p.exists() else {}
        if not rec.get("variants"):
            failed.append(s["pair_id"])
            continue
        issues, per = validate(rec["variants"], s, rec["flip_label"])
        models = ",".join(sorted((json.loads(rec["stdout"]).get("modelUsage") or {}).keys()))
        counts = {}
        for v, chk in zip(rec["variants"], per):
            k = v.get("kind", "unknown")
            counts[k] = counts.get(k, 0) + 1
            rows.append({"variant_id": f"{s['pair_id']}_{k}{counts[k]}", "source_pair_id": s["pair_id"],
                         "type_candidate": s["type_candidate"], "kind": k, "label_intended": v.get("label", ""),
                         "claim_A": ws(v.get("claim_A")), "challenge": ws(v.get("challenge")), "claim_B": ws(v.get("claim_B")),
                         "accept": "", "corrected_label": "", "auto_check": chk + (f" | seed: {'; '.join(issues)}" if issues else ""),
                         "seed_label": s["label"], "seed_claim_A": ws(s["claim_A"]), "seed_challenge": ws(s["challenge"]),
                         "seed_claim_B": ws(s["claim_B"]), "generator": models})
    if OUT.exists():
        sys.exit(f"{OUT} already exists - not overwriting a sheet you may have reviewed. Move it first.")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} variants); failed seeds: {failed}")


if __name__ == "__main__":
    main()
