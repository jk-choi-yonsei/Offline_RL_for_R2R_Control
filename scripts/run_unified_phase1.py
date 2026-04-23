"""
Unified Experiment Phase 1: Sanity Check (48 runs).

Unified settings:
  Reward: r = -α(e/σ_spec)² - λ_s||Δa||₁, α=1.0, λ_s=0.01, clip[-10,0]
  HP: lr=1e-4, cql_alpha=5.0, bc_weight=0.5, batch=256, pretrain=80, CQL=200
  4 datasets × 4 methods × 3 seeds = 48 runs

Results saved to: results/unified_v1/phase1/

Usage:
  python scripts/run_unified_phase1.py                  # run all 48
  python scripts/run_unified_phase1.py --parallel 4     # 4 parallel workers
  python scripts/run_unified_phase1.py --phase 2        # Phase 2 (7 extra seeds)
  python scripts/run_unified_phase1.py --dry-run        # print commands only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "unified_v1"

# Seeds
PHASE1_SEEDS = [123]
PHASE2_SEEDS = [789, 1024, 2024, 7777]
ALL_SEEDS = PHASE1_SEEDS + PHASE2_SEEDS

# Unified HP (shared across all datasets)
UNIFIED_HP = {
    "lr": 1e-4,
    "cql_alpha": 5.0,
    "bc_weight": 0.5,
    "batch_size": 256,
    "pretrain_epochs": 80,
    "epochs": 200,
    "context_dim": 2,
}

# Dataset configurations
DATASETS = {
    "cmp2": {
        "script": "src/rl/train_sarc.py",
        "extra_args": ["--use-wm"],
    },
    "cmp1": {
        "script": "src/rl/train_sarc_cmp1.py",
        "extra_args": ["--chrono-split", "--use-wm"],
    },
    "sim_mild": {
        "script": "src/rl/train_sarc_sim.py",
        "extra_args": ["--drift-scale", "0.4"],
    },
    "sim_medium": {
        "script": "src/rl/train_sarc_sim.py",
        "extra_args": ["--drift-scale", "1.0"],
    },
}


def build_command(dataset: str, seed: int, phase_dir: str) -> list:
    """Build the command for a single experiment run."""
    cfg = DATASETS[dataset]
    cmd = [
        sys.executable, str(PROJECT_ROOT / cfg["script"]),
        "--lr", str(UNIFIED_HP["lr"]),
        "--cql-alpha", str(UNIFIED_HP["cql_alpha"]),
        "--bc-weight", str(UNIFIED_HP["bc_weight"]),
        "--batch-size", str(UNIFIED_HP["batch_size"]),
        "--pretrain-epochs", str(UNIFIED_HP["pretrain_epochs"]),
        "--epochs", str(UNIFIED_HP["epochs"]),
        "--context-dim", str(UNIFIED_HP["context_dim"]),
        "--seed", str(seed),
    ] + cfg["extra_args"]

    # Override run-name / model-suffix for organized output
    tag = f"unified_{dataset}_s{seed}"
    if dataset == "cmp2":
        cmd += ["--run-name", tag]
    elif dataset == "cmp1":
        cmd += ["--model-suffix", f"_unified_s{seed}"]
    # sim uses auto-tag from drift-scale + seed

    return cmd


def run_sequential(commands: list, dry_run: bool = False):
    """Run commands one by one."""
    total = len(commands)
    results = []
    for i, (label, cmd) in enumerate(commands):
        print(f"\n[{i+1}/{total}] {label}")
        print(f"  CMD: {' '.join(cmd)}")
        if dry_run:
            results.append((label, "DRY_RUN", 0))
            continue

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        results.append((label, status, elapsed))
        print(f"  {status} ({elapsed:.0f}s)")
        if proc.returncode != 0:
            print(f"  STDERR: {proc.stderr[-500:]}")

    return results


def _run_one(label_cmd):
    """Run a single experiment (top-level for pickling compatibility)."""
    label, cmd = label_cmd
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    return (label, status, elapsed, proc.stderr[-500:] if proc.returncode != 0 else "")


def run_parallel(commands: list, n_workers: int, dry_run: bool = False):
    """Run commands with N parallel workers (ThreadPool — subprocess-based, no GIL issue)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if dry_run:
        for label, cmd in commands:
            print(f"[DRY] {label}: {' '.join(cmd)}")
        return []

    print(f"Running {len(commands)} experiments with {n_workers} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, lc): lc[0] for lc in commands}
        for fut in as_completed(futures):
            label, status, elapsed, err = fut.result()
            results.append((label, status, elapsed))
            print(f"  [{status}] {label} ({elapsed:.0f}s)", flush=True)
            if err:
                print(f"    STDERR: {err}", flush=True)

    return results


def collect_comparison(phase_dir: str, seeds: list):
    """After experiments, collect results and compare with original."""
    print("\n" + "=" * 70)
    print("COMPARISON: Unified vs Original")
    print("=" * 70)

    # Load original results from memory for reference
    original = {
        "cmp2": {"SARC": 0.235, "BC": 0.246},
        "cmp1": {"SARC": 0.487, "BC": 0.493},
        "sim_mild": {"SARC": 0.125, "BC": 0.167},
        "sim_medium": {"SARC": 0.292, "BC": 0.331},
    }

    # Try to load unified results from JSON files
    results_root = PROJECT_ROOT / "results"
    for dataset in DATASETS:
        print(f"\n--- {dataset.upper()} ---")
        print(f"{'Method':<18} {'Unified MAE':>14} {'Original MAE':>14} {'Delta':>10}")
        print("-" * 58)

        for method in ["SARC", "SARC-no-drift", "BC"]:
            # Collect unified MAEs across seeds
            maes = []
            for seed in seeds:
                # Try different result file patterns
                patterns = [
                    f"sarc_evaluation_unified_{dataset}_s{seed}.json",
                    f"cmp1_evaluation_unified_s{seed}.json",
                    f"sim_evaluation_{dataset}_s{seed}.json",
                ]
                for pat in patterns:
                    fpath = results_root / pat
                    if fpath.exists():
                        with open(fpath) as f:
                            data = json.load(f)
                        if "methods" in data and method in data["methods"]:
                            maes.append(data["methods"][method]["mae"])
                        break

            if maes:
                mean_mae = sum(maes) / len(maes)
                orig = original.get(dataset, {}).get(method, None)
                delta = f"{(mean_mae - orig) / orig * 100:+.1f}%" if orig else "N/A"
                print(f"{method:<18} {mean_mae:>11.4f}±{max(maes)-min(maes):.3f} "
                      f"{orig if orig else 'N/A':>14} {delta:>10}")
            else:
                orig = original.get(dataset, {}).get(method, "N/A")
                print(f"{method:<18} {'(no data)':>14} {orig:>14} {'':>10}")

    # STOP check
    print("\n" + "=" * 70)
    print("STOP CHECK: SARC MAE within ±30% of original?")
    stop = False
    for dataset in DATASETS:
        maes = []
        for seed in seeds:
            for pat in [
                f"sarc_evaluation_unified_{dataset}_s{seed}.json",
                f"cmp1_evaluation_unified_s{seed}.json",
                f"sim_evaluation_{dataset}_s{seed}.json",
            ]:
                fpath = results_root / pat
                if fpath.exists():
                    with open(fpath) as f:
                        data = json.load(f)
                    if "methods" in data and "SARC" in data["methods"]:
                        maes.append(data["methods"]["SARC"]["mae"])
                    break

        if maes:
            mean_mae = sum(maes) / len(maes)
            orig = original.get(dataset, {}).get("SARC", mean_mae)
            pct_change = (mean_mae - orig) / orig * 100
            status = "OK" if abs(pct_change) <= 30 else "STOP"
            if status == "STOP":
                stop = True
            print(f"  {dataset}: {mean_mae:.4f} vs {orig:.4f} ({pct_change:+.1f}%) [{status}]")

    if stop:
        print("\n*** STOP: One or more datasets exceeded ±30% threshold. ***")
        print("*** Review results before proceeding to Phase 2. ***")
    else:
        print("\n*** ALL PASS: Safe to proceed to Phase 2. ***")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run unified experiments")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1 (3 seeds) or Phase 2 (7 extra seeds)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 5 seeds (overrides --phase)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: sequential)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip running, just collect and compare results")
    args = parser.parse_args()

    if args.all:
        seeds = ALL_SEEDS
        phase_dir = "all"
    else:
        seeds = PHASE1_SEEDS if args.phase == 1 else PHASE2_SEEDS
        phase_dir = f"phase{args.phase}"
    out_dir = RESULTS_DIR / phase_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        all_seeds = PHASE1_SEEDS if args.phase == 1 else ALL_SEEDS
        collect_comparison(phase_dir, all_seeds)
        return

    # Build all commands
    commands = []
    for dataset in DATASETS:
        for seed in seeds:
            label = f"{dataset}/s{seed}"
            cmd = build_command(dataset, seed, phase_dir)
            commands.append((label, cmd))

    print(f"Phase {args.phase}: {len(commands)} experiments "
          f"({len(DATASETS)} datasets × {len(seeds)} seeds)")
    print(f"Unified HP: {UNIFIED_HP}")
    print(f"Seeds: {seeds}")
    print()

    # Run
    if args.parallel > 1:
        results = run_parallel(commands, args.parallel, args.dry_run)
    else:
        results = run_sequential(commands, args.dry_run)

    # Summary
    if not args.dry_run:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        n_ok = sum(1 for _, s, _ in results if s == "OK")
        n_fail = len(results) - n_ok
        total_time = sum(t for _, _, t in results)
        print(f"  OK: {n_ok}/{len(results)}, FAIL: {n_fail}, Total: {total_time:.0f}s ({total_time/60:.1f}min)")

        if n_fail > 0:
            print("\n  FAILED:")
            for label, status, _ in results:
                if status != "OK":
                    print(f"    {label}: {status}")

        # Auto-compare
        collect_comparison(phase_dir, seeds)

    # Save run log
    log_path = out_dir / "run_log.json"
    log = {
        "phase": args.phase,
        "seeds": seeds,
        "hp": UNIFIED_HP,
        "results": [(l, s, t) for l, s, t in results] if not args.dry_run else "dry_run",
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nRun log saved: {log_path}")


if __name__ == "__main__":
    main()
