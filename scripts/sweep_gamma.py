"""
Instant gamma-sweep over the stored baseline grid (results/e6_grid_*.json). For each
gamma, selects every baseline by val (MAE + gamma*CE), then reports its TEST
(MAE, CE) + one-sided Wilcoxon vs SARC + point-dominance, plus the criterion-free
baseline Pareto frontier (does SARC dominate the whole front?). No re-evaluation,
so sweep as many gammas as you like.

Usage:
  python scripts/sweep_gamma.py --gammas 0,0.05,0.1,0.3,1.0 --box 1.0
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(_HERE), "results")
TESTBEDS = ["cmp2", "cmp1", "sim_mild", "sim_medium"]
BL_ORDER = ["D-EWMA", "age_dEWMA", "Kalman"]


def wilc2(s, b):
    """Two-sided Wilcoxon + direction (who has the lower metric)."""
    s, b = np.array(s), np.array(b)
    try:
        p = float(wilcoxon(s, b, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        p = 1.0
    return ("SARC" if s.mean() < b.mean() else "base"), p


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def mark(winner, p):
    """S*** = SARC sig lower; B*** = baseline sig lower; ns = no sig difference."""
    s = sig(p)
    return "ns" if s == "ns" else f"{'S' if winner == 'SARC' else 'B'}{s}"


def pareto_front(pts):
    P = sorted(pts, key=lambda q: (q[0], q[1]))
    fr, best = [], float("inf")
    for mae, ce in P:
        if ce < best - 1e-12:
            fr.append((mae, ce)); best = ce
    return fr


def select(cfgs, gamma):
    return min(cfgs, key=lambda c: c["val_mae"] + gamma * c["val_ce"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", default="0,0.1,0.3,1.0")
    ap.add_argument("--box", default="1.0")
    a = ap.parse_args()
    gammas = [float(g) for g in a.gammas.split(",")]
    bkey = f"box_{a.box}"

    data = {}
    for tb in TESTBEDS:
        p = os.path.join(RES, f"e6_grid_{tb}.json")
        if os.path.exists(p):
            data[tb] = json.load(open(p))
    if not data:
        print("no e6_grid_*.json found in results/"); return

    for gamma in gammas:
        print(f"\n{'='*94}\n gamma = {gamma}   (baseline selected by val MAE + {gamma}*CE ; box {a.box})\n{'='*94}")
        print(f"{'testbed':<11}{'SARC MAE/CE':<15}{'baseline':<16}{'sel MAE/CE':<14}"
              f"{'MAE':<8}{'CE':<8}{'dom?':<9}   (two-sided: S*=SARC sig lower, B*=baseline sig lower)")
        for tb in TESTBEDS:
            if tb not in data:
                continue
            box = data[tb]["boxes"][bkey]
            S = box["SARC"]; s_mae, s_ce = S["test_mae"], S["test_ce"]
            all_pts, first = [], True
            for name in BL_ORDER:
                cfgs = box["baselines"][name]
                all_pts += [(c["test_mae"], c["test_ce"]) for c in cfgs]
                sel = select(cfgs, gamma)
                wm, pm = wilc2(S["test_ps_mae"], sel["test_ps_mae"])
                wc, pc = wilc2(S["test_ps_ce"], sel["test_ps_ce"])
                dom = (s_mae <= sel["test_mae"] + 1e-9) and (s_ce <= sel["test_ce"] + 1e-9)
                head = f"{tb:<11}{s_mae:.3f}/{s_ce:.3f}    " if first else " " * 26
                first = False
                print(f"{head}{name:<16}{sel['test_mae']:.3f}/{sel['test_ce']:.3f}   "
                      f"{mark(wm, pm):<8}{mark(wc, pc):<8}{str(dom):<9}")
            fr = pareto_front(all_pts)
            dom_front = all((s_mae <= m + 1e-9 and s_ce <= c + 1e-9) for m, c in fr)
            print(f"{'':<26}=> SARC dominates full baseline Pareto front? {dom_front}   "
                  f"front={', '.join(f'({m:.2f},{c:.2f})' for m, c in fr)}")


if __name__ == "__main__":
    main()
