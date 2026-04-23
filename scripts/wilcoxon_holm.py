"""
Sequence-level paired Wilcoxon signed-rank tests + Holm correction.

Reads the per-sequence MAE/CE arrays emitted by the patched RolloutEvaluator
(fields: per_sequence_mae, per_sequence_ce) and reports, for each testbed,
whether SARC's per-sequence metric vector is significantly lower than each
baseline's.

Deterministic baselines (D-EWMA, Kalman) have identical per-sequence
vectors across seeds, so we use any one seed. Stochastic methods
(SARC, BC) are averaged per-sequence across 5 seeds before testing --
seed variance is absorbed, sequence variance drives the test (n = n_sequences).

Holm correction is applied within each (testbed, metric) family over the
set of baseline comparisons.

For CMP1, Stage B MAE is additionally tested using per_stage.B.per_sequence_mae.

Usage:
    python scripts/wilcoxon_holm.py
    python scripts/wilcoxon_holm.py --latex    # print LaTeX-ready marker table
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"

SEEDS = [123, 789, 1024, 2024, 7777]

TESTBEDS = {
    "Sim Mild":   [RESULTS / f"sim_evaluation_sim_mild_cql1.0_s{s}.json"   for s in SEEDS],
    "Sim Medium": [RESULTS / f"sim_evaluation_sim_medium_cql1.0_s{s}.json" for s in SEEDS],
    "CMP1":       [RESULTS / f"cmp1_evaluation_unified_s{s}_s{s}.json"     for s in SEEDS],
    "CMP2":       [RESULTS / f"sarc_evaluation_final_cmp2_cql1.0_s{s}.json" for s in SEEDS],
}

BASELINES = ["SARC-no-drift", "BC", "D-EWMA", "Kalman"]
METRICS = [("MAE", "per_sequence_mae"), ("CE", "per_sequence_ce")]

# Extra per-stage metrics tested only for specific testbeds.
# Format: testbed_name -> list of (display_name, stage_key, metric_key)
STAGE_METRICS = {
    "CMP1": [("Stage B MAE", "B", "per_sequence_mae")],
}


def load_per_sequence(
    files: List[Path],
    method: str,
    key: str,
    stage: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Return (n_seeds, n_sequences) array, or None if missing.

    If stage is set (e.g. 'B'), reads from per_stage[stage][method][key].
    """
    vecs = []
    for f in files:
        if not f.exists():
            continue
        with open(f) as fp:
            data = json.load(fp)
        if stage is not None:
            container = data.get("per_stage", {}).get(stage, {}).get(method)
        else:
            container = data.get("methods", {}).get(method)
        if container is None or key not in container:
            return None
        vecs.append(np.asarray(container[key], dtype=float))
    if not vecs:
        return None
    lengths = {len(v) for v in vecs}
    if len(lengths) != 1:
        return None
    return np.vstack(vecs)  # (n_seeds, n_seq)


def holm_adjust(pvals: List[float]) -> List[float]:
    """Return Holm-adjusted p-values preserving input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj.tolist()


def sig_marker(p_adj: Optional[float]) -> str:
    if p_adj is None:
        return ""
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return ""


def latex_marker(p_adj: Optional[float], status: str) -> str:
    if status != "ok" or p_adj is None:
        return "?"
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return "–"


def run_metric_tests(
    files: List[Path],
    sarc_vec: np.ndarray,
    metric_name: str,
    key: str,
    stage: Optional[str] = None,
) -> Tuple[List, List, List]:
    """Return (valid_baselines, raw_ps, diffs_mean) for one metric."""
    raw_ps, diffs_mean, valid_baselines = [], [], []
    missing_rows = []
    for base in BASELINES:
        arr = load_per_sequence(files, base, key, stage=stage)
        if arr is None:
            missing_rows.append((metric_name, base, None, None, None, "missing"))
            continue
        base_vec = arr.mean(axis=0)
        if base_vec.shape != sarc_vec.shape:
            missing_rows.append((metric_name, base, None, None, None, "shape mismatch"))
            continue
        try:
            stat = wilcoxon(sarc_vec, base_vec, alternative="less", zero_method="wilcox")
            p = float(stat.pvalue)
        except ValueError:
            p = 1.0
        raw_ps.append(p)
        diffs_mean.append(float(np.mean(sarc_vec - base_vec)))
        valid_baselines.append(base)
    return valid_baselines, raw_ps, diffs_mean, missing_rows


def analyze_testbed(name: str, files: List[Path]) -> Dict:
    existing = [f for f in files if f.exists()]
    if not existing:
        return {"error": "no result files found"}

    # Build list of all (display_name, key, stage) to test
    all_metrics: List[Tuple[str, str, Optional[str]]] = [
        (mn, key, None) for mn, key in METRICS
    ]
    for display, stage, key in STAGE_METRICS.get(name, []):
        all_metrics.append((display, key, stage))

    # Load SARC seed-averaged vectors for each metric
    sarc_vecs: Dict[str, np.ndarray] = {}
    for display, key, stage in all_metrics:
        arr = load_per_sequence(files, "SARC", key, stage=stage)
        if arr is None:
            return {"error": f"SARC missing {display}"}
        sarc_vecs[display] = arr.mean(axis=0)

    rows = []
    for display, key, stage in all_metrics:
        sarc_vec = sarc_vecs[display]
        valid_bases, raw_ps, diffs, missing = run_metric_tests(
            files, sarc_vec, display, key, stage=stage
        )
        rows.extend(missing)
        adj_ps = holm_adjust(raw_ps) if raw_ps else []
        for base, p_raw, p_adj, diff in zip(valid_bases, raw_ps, adj_ps, diffs):
            rows.append((display, base, p_raw, p_adj, diff, "ok"))

    n_seq = int(len(sarc_vecs["MAE"]))
    extra = {
        f"n_seq_{d.replace(' ', '_')}": int(len(v))
        for d, v in sarc_vecs.items()
        if d != "MAE"
    }
    return {"n_sequences": n_seq, **extra, "rows": rows}


def print_report(results: Dict, latex: bool):
    for name, res in results.items():
        print("=" * 72)
        print(name)
        print("=" * 72)
        if "error" in res:
            print(f"  {res['error']}")
            continue
        print(f"  n_sequences = {res['n_sequences']}")
        print(f"  {'metric':<14}{'baseline':<10}{'Δ(SARC-base)':>14}{'p_raw':>12}{'p_Holm':>12}  sig")
        for metric, base, p_raw, p_adj, diff, status in res["rows"]:
            if status != "ok":
                print(f"  {metric:<14}{base:<10}{'--':>14}{'--':>12}{'--':>12}  ({status})")
                continue
            sig = sig_marker(p_adj)
            print(f"  {metric:<14}{base:<10}{diff:>+14.5f}{p_raw:>12.4g}{p_adj:>12.4g}  {sig}")

    if latex:
        # Collect all metric display names across testbeds
        all_metric_names: List[str] = []
        for mn, _ in METRICS:
            if mn not in all_metric_names:
                all_metric_names.append(mn)
        for name in results:
            for display, _, _ in STAGE_METRICS.get(name, []):
                if display not in all_metric_names:
                    all_metric_names.append(display)

        print("\n" + "=" * 72)
        print("LaTeX significance markers (SARC significantly LOWER than baseline)")
        print("*: p<0.05  **: p<0.01  ***: p<0.001  (Holm-adjusted, one-sided)")
        print("=" * 72)
        header = f"  {'Testbed':<12}{'Metric':<16}" + "".join(f"{b:>9}" for b in BASELINES)
        print(header)
        for name, res in results.items():
            if "error" in res:
                continue
            by = {(m, b): (p_adj, status) for (m, b, _, p_adj, _, status) in res["rows"]}
            # Which metrics exist for this testbed
            testbed_metrics = [mn for mn, _ in METRICS] + [
                d for d, _, _ in STAGE_METRICS.get(name, [])
            ]
            for metric in testbed_metrics:
                marks = [latex_marker(*by.get((metric, b), (None, "missing")))
                         for b in BASELINES]
                print(f"  {name:<12}{metric:<16}" + "".join(f"{m:>9}" for m in marks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    results = {name: analyze_testbed(name, files) for name, files in TESTBEDS.items()}
    print_report(results, args.latex)


if __name__ == "__main__":
    main()
