#!/usr/bin/env bash
# End-to-end reproduction of the CMP1 (PHM 2016) experiments reported in the SARC paper (Table III).
#
# Pipeline:
#   1. Preprocess raw CMP1 CSVs into the R2R MDP format (chronological 70/15/15 split).
#   2. Train the Neural World Model (5-ensemble) on CMP1.
#   3. Run the unified Phase 1 + Phase 2 evaluation across 5 seeds for SARC, SARC-no-drift, BC, D-EWMA, Kalman.
#   4. Aggregate 5-seed statistics and run Wilcoxon signed-rank with Holm correction.
#   5. Emit final LaTeX-style result tables.
#
# Expected wall-clock: ~3-6 h on a single modern GPU (CUDA), ~1 day on CPU.
#
# All intermediate artifacts are written under `results/`. The script is idempotent — partial reruns will
# reuse existing checkpoints when present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -d "Dataset/CMP1/CMP-data/training" || ! -d "Dataset/CMP1/CMP-data/test" ]]; then
    echo "ERROR: Dataset/CMP1/CMP-data/{training,test}/ not found." >&2
    echo "       The PHM 2016 mirror should be bundled at Dataset/CMP1/. See Dataset/CMP1/README.md." >&2
    exit 1
fi

echo "[1/5] Preprocessing CMP1 raw CSVs into R2R MDP arrays..."
python -m src.data.preprocess_cmp1_r2r

echo "[2/5] Training Neural World Model (5-ensemble) on CMP1..."
python -m src.models.train_world_model_cmp1

echo "[3/5] Running 5-seed evaluation for SARC, SARC-no-drift, BC, D-EWMA, Kalman..."
python scripts/run_unified_phase1.py
python scripts/run_unified_phase1.py --phase 2

echo "[4/5] Aggregating 5-seed statistics..."
python scripts/aggregate_5seed.py
python scripts/wilcoxon_holm.py

echo "[5/5] Generating final result tables..."
python scripts/final_statistics.py

echo
echo "Done. Compare results/summary_5seed.csv against Table III in the SARC paper."
