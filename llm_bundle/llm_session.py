"""Unattended LLM session (vLLM): smoke-test, then open-weight models in order, then NLI stage 2.

Each step runs as a subprocess with its own timeout; the whole session has a hard deadline (--hours).
A step whose smoke test failed is skipped in the full pass. A failed or timed-out step does not stop the
session. Logs: results/logs/<step>.log; overview: results/session_summary.json (also printed).

Usage (on the instance, via jl run):  python llm_session.py [--hours 3]
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
Q7, Q32 = "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-32B-Instruct"
SMOKE = [
    ("smoke_qwen7b", [PY, "llm_run.py", "--model", Q7, "--smoke"], 30),
    ("smoke_stage2", [PY, "nli_stage2.py", "--smoke"], 15),
]
FULL = [  # (name, command, timeout minutes, smoke dependency)
    ("qwen7b", [PY, "llm_run.py", "--model", Q7], 30, "smoke_qwen7b"),
    ("qwen32b", [PY, "llm_run.py", "--model", Q32], 75, "smoke_qwen7b"),
    ("stage2", [PY, "nli_stage2.py"], 20, "smoke_stage2"),
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
    ap.add_argument("--hours", type=float, default=3.0)
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
