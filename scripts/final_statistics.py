"""
Final paper statistics: unified reward + dataset-specific cql_alpha.

Sources:
  CMP2:       results/sarc_evaluation_final_cmp2_cql1.0_s*.json  (cql_alpha=1.0)
  CMP1:       results/cmp1_evaluation_unified_s*_s*.json          (cql_alpha=5.0)
  Sim Mild:   results/sim_evaluation_sim_mild_s*.json             (cql_alpha=5.0)
  Sim Medium: results/sim_evaluation_sim_medium_s*.json           (cql_alpha=5.0)
"""

import json
import glob
import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = "results"

PATTERNS = {
    "CMP2": "sarc_evaluation_final_cmp2_cql1.0_s*.json",
    "CMP1": "cmp1_evaluation_unified_s*_s*.json",
    "Sim Mild": "sim_evaluation_sim_mild_s*.json",
    "Sim Medium": "sim_evaluation_sim_medium_s*.json",
}

# Original paper results for comparison
ORIGINAL = {
    "CMP2": {"SARC": 0.235, "SARC-no-drift": 0.234, "BC": 0.246},
    "CMP1": {"SARC": 0.487, "SARC-no-drift": 0.488, "BC": 0.493},
    "Sim Mild": {"SARC": 0.125, "SARC-no-drift": 0.372, "BC": 0.167},
    "Sim Medium": {"SARC": 0.292, "SARC-no-drift": 0.279, "BC": 0.331},
}

ALL_METHODS = ["SARC", "SARC-no-drift", "BC", "D-EWMA", "Kalman"]


def load_results(pattern):
    files = sorted(glob.glob(f"{RESULTS_DIR}/{pattern}"))
    maes, costs = {}, {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        for method, vals in d.get("methods", {}).items():
            maes.setdefault(method, []).append(vals["mae"])
            costs.setdefault(method, []).append(vals["action_cost"])
    return maes, costs, len(files)


def wilcoxon_one_sided(a, b):
    """H0: a >= b, H1: a < b (SARC better = lower MAE)."""
    a, b = np.array(a), np.array(b)
    n = min(len(a), len(b))
    if n < 5:
        return float("nan"), n
    a, b = a[:n], b[:n]
    diff = a - b
    if np.all(diff == 0):
        return 1.0, n
    try:
        _, p = wilcoxon(a, b, alternative="less")
        return p, n
    except Exception:
        return float("nan"), n


def sig_str(p):
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    print("=" * 90)
    print("FINAL PAPER RESULTS: Unified Reward + Dataset-Specific cql_alpha")
    print("  Reward: r = -α(e/σ_spec)² - λ_s||Δa||₁  (α=1, λ_s=0.01)")
    print("  HP: lr=1e-4, bc=0.5, batch=256, pretrain=80, CQL=200, ctx=2")
    print("  CMP2: cql_alpha=1.0  |  CMP1/Sim: cql_alpha=5.0")
    print("=" * 90)

    for dataset in ["CMP2", "CMP1", "Sim Mild", "Sim Medium"]:
        maes, costs, n_seeds = load_results(PATTERNS[dataset])

        print(f"\n{'─' * 90}")
        print(f"  {dataset} ({n_seeds} seeds)")
        print(f"{'─' * 90}")

        # Table
        print(f"  {'Method':<18} {'MAE (mean±std)':>18} {'Act.Cost (mean±std)':>22}")
        print(f"  {'-' * 60}")
        for m in ALL_METHODS:
            if m in maes:
                ma = np.array(maes[m])
                ca = np.array(costs.get(m, [0]))
                print(f"  {m:<18} {np.mean(ma):.4f} ± {np.std(ma):.4f}   "
                      f"{np.mean(ca):.4f} ± {np.std(ca):.4f}")

        # Wilcoxon
        if "SARC" not in maes:
            continue

        sarc = maes["SARC"]
        print(f"\n  Wilcoxon signed-rank (one-sided: SARC < other):")
        print(f"  {'Comparison':<28} {'p-value':>10} {'n':>4} {'Sig':>6} {'Δ%':>8}")
        print(f"  {'-' * 60}")

        for other in ["BC", "SARC-no-drift", "D-EWMA", "Kalman"]:
            if other not in maes:
                continue
            p, n = wilcoxon_one_sided(sarc, maes[other])
            s_mean = np.mean(sarc[:n])
            o_mean = np.mean(np.array(maes[other])[:n])
            delta = (o_mean - s_mean) / o_mean * 100 if o_mean > 0 else 0
            print(f"  SARC vs {other:<18} {p:>10.4f} {n:>4} {sig_str(p):>6} {delta:>+7.1f}%")

    # Comparison with original
    print(f"\n{'=' * 90}")
    print("  vs ORIGINAL PAPER (dataset-specific HP)")
    print(f"{'=' * 90}")
    print(f"  {'Dataset':<12} {'Method':<16} {'Original':>10} {'Final':>10} {'Δ%':>8}")
    print(f"  {'-' * 60}")
    for dataset in ["CMP2", "CMP1", "Sim Mild", "Sim Medium"]:
        maes, _, _ = load_results(PATTERNS[dataset])
        for method in ["SARC", "BC"]:
            if method in maes and method in ORIGINAL.get(dataset, {}):
                f_mean = np.mean(maes[method])
                o_mean = ORIGINAL[dataset][method]
                delta = (f_mean - o_mean) / o_mean * 100
                print(f"  {dataset:<12} {method:<16} {o_mean:>10.4f} {f_mean:>10.4f} {delta:>+7.1f}%")

    # Action cost comparison vs D-EWMA
    print(f"\n{'=' * 90}")
    print("  ACTION COST ADVANTAGE (SARC vs D-EWMA)")
    print(f"{'=' * 90}")
    for dataset in ["CMP2", "CMP1", "Sim Mild", "Sim Medium"]:
        _, costs, _ = load_results(PATTERNS[dataset])
        if "SARC" in costs and "D-EWMA" in costs:
            s_cost = np.mean(costs["SARC"])
            d_cost = np.mean(costs["D-EWMA"])
            ratio = d_cost / s_cost if s_cost > 0 else float("inf")
            print(f"  {dataset:<12}  SARC={s_cost:.4f}  D-EWMA={d_cost:.4f}  "
                  f"ratio={ratio:.1f}x  ({(1-s_cost/d_cost)*100:.0f}% lower)")


if __name__ == "__main__":
    main()
