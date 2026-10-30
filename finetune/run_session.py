"""Unattended GPU session: smoke-test every step, then run the full steps in priority order.

Each step runs as a subprocess with its own timeout; the whole session has a hard deadline (--hours).
A step whose smoke test failed is skipped in the full pass. A failed or timed-out step does not stop the
session. Logs: results/logs/<step>.log; overview: results/session_summary.json (also printed).

Usage (on the instance, via jl run):  python run_session.py [--hours 8] [--only nli_pair,probing]
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
DATA = "data/dataset_augmented_all_v2.csv"
TRAIN = [PY, "train_cv.py", "--data", DATA]

SMOKE = [
    ("smoke_features", [PY, "features_gpu.py", "--data", DATA, "--smoke"], 10),
    ("smoke_probing", [PY, "probe_layers.py", "--data", DATA, "--smoke", "--models", "deberta_nli"], 15),
    ("smoke_train", TRAIN + ["--smoke", "--tag", "session"], 10),
    ("smoke_roberta", TRAIN + ["--smoke", "--model", "FacebookAI/roberta-large-mnli", "--tag", "session_roberta"], 10),
]
FULL = [  # (name, command, timeout minutes, which smoke test it depends on) - most important first
    ("features", [PY, "features_gpu.py", "--data", DATA, "--check", "data/dataset_consolidated_v2_nli_features.csv"], 20, "smoke_features"),
    ("nli_pair", TRAIN + ["--init", "nli", "--seeds", "0,1,2,3,4", "--tag", "nli_pair"], 150, "smoke_train"),
    ("probing", [PY, "probe_layers.py", "--data", DATA], 45, "smoke_probing"),
    ("nli_bonly", TRAIN + ["--init", "nli", "--input", "b_only", "--tag", "nli_bonly"], 60, "smoke_train"),
    ("plain_pair", TRAIN + ["--init", "plain", "--tag", "plain_pair"], 90, "smoke_train"),
    ("roberta_mnli_pair", TRAIN + ["--model", "FacebookAI/roberta-large-mnli", "--tag", "roberta_mnli_pair"], 90, "smoke_roberta"),
    ("nli_pair_lr1e-5", TRAIN + ["--lr", "1e-5", "--tag", "nli_pair_lr1e-5"], 90, "smoke_train"),
    ("nli_pair_lr3e-5", TRAIN + ["--lr", "3e-5", "--tag", "nli_pair_lr3e-5"], 90, "smoke_train"),
    ("nli_pair_frac0.25", TRAIN + ["--train_frac", "0.25", "--tag", "nli_pair_frac0.25"], 60, "smoke_train"),
    ("nli_pair_frac0.5", TRAIN + ["--train_frac", "0.5", "--tag", "nli_pair_frac0.5"], 60, "smoke_train"),
    ("nli_pair_frac0.75", TRAIN + ["--train_frac", "0.75", "--tag", "nli_pair_frac0.75"], 75, "smoke_train"),
]


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def run(name, cmd, timeout_min, log_dir):
    log = log_dir / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_min * 60).returncode
            status = "ok" if rc == 0 else f"failed (exit {rc})"
        except subprocess.TimeoutExpired:
            status = f"timed out after {timeout_min} min"
    tail = log.read_text().strip().splitlines()[-3:]
    return dict(step=name, status=status, minutes=round((time.time() - t0) / 60, 1), log=str(log.relative_to(HERE)),
                last_lines=tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--only", default="", help="comma-separated full-step names to run (default: all)")
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    log_dir = HERE / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {"start": now(), "hours_limit": args.hours, "steps": []}
    only = set(args.only.split(",")) if args.only else None

    def save():
        (HERE / "results" / "session_summary.json").write_text(json.dumps(summary, indent=2))

    smoke_ok = {}
    for name, cmd, tmo in SMOKE:
        r = run(name, cmd, tmo, log_dir)
        smoke_ok[name] = r["status"] == "ok"
        summary["steps"].append(r)
        print(f"{now()}  {name:22s} {r['status']:28s} {r['minutes']:6.1f} min", flush=True)
        save()

    for name, cmd, tmo, needs in FULL:
        if only and name not in only:
            continue
        left_min = (deadline - time.time()) / 60
        if not smoke_ok.get(needs, False):
            r = dict(step=name, status=f"skipped ({needs} failed)", minutes=0)
        elif left_min < 5:
            r = dict(step=name, status="skipped (session deadline reached)", minutes=0)
        else:
            r = run(name, cmd, min(tmo, int(left_min)), log_dir)
        summary["steps"].append(r)
        print(f"{now()}  {name:22s} {r['status']:28s} {r['minutes']:6.1f} min", flush=True)
        save()

    summary["end"] = now()
    save()
    print("\n=== session summary ===", flush=True)
    for r in summary["steps"]:
        print(f"  {r['step']:22s} {r['status']}", flush=True)
    print("RESULTS_DIR=" + str(HERE / "results"), flush=True)


if __name__ == "__main__":
    main()
