"""
CMP2 final 5-seed experiment: unified reward + cql_alpha=1.0.

Unified reward: r = -α(e/σ_spec)² - λ_s||Δa||₁, α=1.0, λ_s=0.01
HP: lr=1e-4, cql_alpha=1.0, bc_weight=0.5, batch=256, pretrain=80, CQL=200
5 seeds × 1 run (SARC+BC+SARC-no-drift all in one) = 5 runs

Usage:
  python scripts/run_cmp2_final.py --parallel 4
  python scripts/run_cmp2_final.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = [123, 789, 1024, 2024, 7777]


def build_commands():
    py = sys.executable
    commands = []
    for seed in SEEDS:
        tag = f"final_cmp2_cql1.0_s{seed}"
        commands.append((
            f"cmp2/s{seed}",
            [py, "src/rl/train_sarc.py",
             "--cql-alpha", "1.0",
             "--bc-weight", "0.5",
             "--lambda-s", "0.01",
             "--lr", "1e-4",
             "--batch-size", "256",
             "--pretrain-epochs", "80",
             "--epochs", "200",
             "--context-dim", "2",
             "--seed", str(seed),
             "--use-wm",
             "--run-name", tag],
        ))
    return commands


def _run_one(label_cmd):
    label, cmd = label_cmd
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    stderr_tail = proc.stderr[-500:] if proc.returncode != 0 else ""
    return label, status, elapsed, stderr_tail


def main():
    parser = argparse.ArgumentParser(description="CMP2 final 5-seed (cql_alpha=1.0)")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = build_commands()
    print(f"CMP2 final: {len(commands)} runs (cql_alpha=1.0, unified reward)")
    print(f"Seeds: {SEEDS}\n")

    if args.dry_run:
        for label, cmd in commands:
            print(f"[DRY] {label}: {' '.join(cmd)}")
        return

    results = []
    if args.parallel > 1:
        print(f"Running with {args.parallel} parallel workers...")
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_run_one, lc): lc[0] for lc in commands}
            for fut in as_completed(futures):
                label, status, elapsed, err = fut.result()
                results.append((label, status, elapsed))
                print(f"  [{status}] {label} ({elapsed:.0f}s)", flush=True)
                if err:
                    print(f"    STDERR: {err}", flush=True)
    else:
        for i, (label, cmd) in enumerate(commands):
            print(f"\n[{i+1}/{len(commands)}] {label}")
            label, status, elapsed, err = _run_one((label, cmd))
            results.append((label, status, elapsed))
            print(f"  {status} ({elapsed:.0f}s)")
            if err:
                print(f"  STDERR: {err}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_ok = sum(1 for _, s, _ in results if s == "OK")
    n_fail = len(results) - n_ok
    total_time = sum(t for _, _, t in results)
    print(f"  OK: {n_ok}/{len(results)}, FAIL: {n_fail}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    if n_fail > 0:
        print("\n  FAILED:")
        for label, status, _ in results:
            if status != "OK":
                print(f"    {label}: {status}")


if __name__ == "__main__":
    main()
