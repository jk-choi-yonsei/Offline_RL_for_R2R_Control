"""
Aggregate 5-seed final results for the paper: CMP2, CMP1, Sim-Mild, Sim-Medium.
Reads JSONs in results/ for seeds [123, 789, 1024, 2024, 7777], computes
mean+/-std per method+metric, runs Wilcoxon signed-rank (one-sided SARC<other),
and writes results/summary_5seed.csv and results/summary_5seed.json.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

SEEDS = [123, 789, 1024, 2024, 7777]
METHODS = ["SARC", "SARC-no-drift", "BC", "D-EWMA", "Kalman"]
METRICS = ["mae", "action_cost"]

DATASETS = {
    "CMP2":       "sarc_evaluation_final_cmp2_cql1.0_s{seed}.json",
    "CMP1":       "cmp1_evaluation_unified_s{seed}_s{seed}.json",
    "Sim-Mild":   "sim_evaluation_sim_mild_cql1.0_s{seed}.json",
    "Sim-Medium": "sim_evaluation_sim_medium_cql1.0_s{seed}.json",
}


def load(dataset):
    rows = []
    for s in SEEDS:
        path = os.path.join(RESULTS_DIR, DATASETS[dataset].format(seed=s))
        if not os.path.exists(path):
            print(f"  [MISS] {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        methods = d.get("methods", d)
        per_method = {}
        for m in METHODS:
            if m in methods:
                per_method[m] = {k: methods[m].get(k) for k in METRICS}
        extras = {}
        if "per_stage" in d and "B" in d["per_stage"]:
            extras["StageB"] = {m: d["per_stage"]["B"].get(m, {}) for m in METHODS}
        rows.append({"seed": s, "methods": per_method, "extras": extras})
    return rows


def summary(rows):
    out = {}
    for m in METHODS:
        entry = {}
        for metric in METRICS:
            vals = [r["methods"].get(m, {}).get(metric) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                entry[metric] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals, ddof=0)),
                    "values": vals,
                }
        if entry:
            out[m] = entry
    return out


def wilcoxon(rows, other):
    """One-sided: SARC < other (each metric)."""
    out = {}
    for metric in METRICS:
        sarc = [r["methods"]["SARC"][metric] for r in rows if "SARC" in r["methods"]]
        oth  = [r["methods"].get(other, {}).get(metric) for r in rows]
        oth  = [v for v in oth if v is not None]
        if len(sarc) < 3 or len(oth) != len(sarc):
            continue
        if len(set(oth)) == 1:
            out[metric] = {"deterministic": True, "delta": float(np.mean(sarc) - oth[0])}
            continue
        try:
            stat, p = stats.wilcoxon(sarc, oth, alternative="less")
            out[metric] = {"p": float(p), "sarc_mean": float(np.mean(sarc)),
                           "other_mean": float(np.mean(oth))}
        except Exception as e:
            out[metric] = {"error": str(e)}
    return out


def main():
    all_out = {}
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        rows = load(ds)
        print(f"  loaded {len(rows)}/5 seeds")
        if not rows:
            continue
        s = summary(rows)
        tests = {}
        for m in METHODS:
            if m == "SARC":
                continue
            tests[m] = wilcoxon(rows, m)
        all_out[ds] = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows],
                       "summary": s, "wilcoxon_SARC_less": tests}

        # Human-readable per-dataset
        print(f"  {'Method':<16} {'MAE (mean+/-std)':<22} {'Act.Cost':<22}")
        for m in METHODS:
            if m not in s:
                continue
            mae = s[m].get("mae", {})
            ac  = s[m].get("action_cost", {})
            mae_str = f"{mae['mean']:.4f}+/-{mae['std']:.4f}" if mae else "-"
            ac_str  = f"{ac['mean']:.4f}+/-{ac['std']:.4f}"  if ac else "-"
            print(f"  {m:<16} {mae_str:<22} {ac_str:<22}")
        print(f"  Wilcoxon (SARC<other, one-sided):")
        for m, t in tests.items():
            if t:
                for metric, r in t.items():
                    if "p" in r:
                        print(f"    vs {m:<15} {metric}: p={r['p']:.4f}")
                    elif "deterministic" in r:
                        print(f"    vs {m:<15} {metric}: det (d={r['delta']:+.4f})")

    # Save
    out_json = os.path.join(RESULTS_DIR, "summary_5seed.json")
    with open(out_json, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {out_json}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "summary_5seed.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,method,mae_mean,mae_std,ac_mean,ac_std,n_seeds\n")
        for ds, dd in all_out.items():
            for m, ms in dd["summary"].items():
                mae = ms.get("mae", {})
                ac  = ms.get("action_cost", {})
                f.write(f"{ds},{m},{mae.get('mean','')},{mae.get('std','')},"
                        f"{ac.get('mean','')},{ac.get('std','')},{dd['n_seeds']}\n")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
