"""
CMP1 R2R Control Preprocessing Pipeline.

Converts CMP1 (public PHM dataset) into MDP tuples for R2R control.

Data structure:
  - One file = 1 LOT (multiple wafers)
  - Each row = one timestamp during polishing (~111 rows per wafer)
  - (WAFER_ID, STAGE, CHAMBER) = one polishing run
  - R2R sequence: consecutive wafers in chronological order within the same
    (lot_file, STAGE, CHAMBER)

State (13D):
  - Wear (6): last value of USAGE_OF_* (cumulative usage)
  - Process (6): mean slurry/rotation values
  - prev_RR (1): removal rate from the previous run

Action (6D): pressure setpoints applied to run t+1 (R2R control decision)
  - PRESSURIZED_CHAMBER, MAIN_OUTER_AIR_BAG, CENTER_AIR_BAG,
    RETAINER_RING, RIPPLE_AIR_BAG, EDGE_AIR_BAG pressure (mean of run t+1)
  -> reward = f(RR_{t+1}): outcome of that action

Drift (13D):
  - Wear (6): absolute values
  - Delta wear (6): change relative to previous run
  - lot_position (1): position within lot, normalized to [0, 1]
"""

import os
import glob
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.config import CMP1_DIR, DataConfig

logger = logging.getLogger(__name__)

# ============================================================
# Column Definitions
# ============================================================

WEAR_COLS = [
    "USAGE_OF_BACKING_FILM",
    "USAGE_OF_DRESSER",
    "USAGE_OF_POLISHING_TABLE",
    "USAGE_OF_DRESSER_TABLE",
    "USAGE_OF_MEMBRANE",
    "USAGE_OF_PRESSURIZED_SHEET",
]

PRESSURE_COLS = [
    "PRESSURIZED_CHAMBER_PRESSURE",
    "MAIN_OUTER_AIR_BAG_PRESSURE",
    "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE",
    "RIPPLE_AIR_BAG_PRESSURE",
    "EDGE_AIR_BAG_PRESSURE",
]

PROCESS_COLS = [
    "SLURRY_FLOW_LINE_A",
    "SLURRY_FLOW_LINE_B",
    "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION",
    "STAGE_ROTATION",
    "HEAD_ROTATION",
]

RR_OUTLIER_THRESHOLD = 500.0  # exclude AVG_REMOVAL_RATE > 500


# ============================================================
# Step 1: Aggregate each lot file by run
# ============================================================

def aggregate_runs_from_file(filepath: str, file_id: int) -> pd.DataFrame:
    """
    Aggregate one lot file into per-run rows grouped by (WAFER_ID, STAGE, CHAMBER).

    Returns:
        DataFrame where each row is one polishing run.
        Columns: wear_last, pressure_mean, process_mean, first_timestamp, ...
    """
    df = pd.read_csv(filepath)

    # numeric conversion
    for col in WEAR_COLS + PRESSURE_COLS + PROCESS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    records = []
    for (wafer_id, stage, chamber), group in df.groupby(["WAFER_ID", "STAGE", "CHAMBER"]):
        group = group.sort_values("TIMESTAMP")
        rec = {
            "file_id": file_id,
            "WAFER_ID": wafer_id,
            "STAGE": stage,
            "CHAMBER": chamber,
            "first_timestamp": group["TIMESTAMP"].iloc[0],
        }
        # Wear: last value (cumulative usage)
        for col in WEAR_COLS:
            if col in group.columns:
                rec[f"{col}_last"] = group[col].dropna().iloc[-1] if group[col].notna().any() else np.nan

        # Pressure & Process: mean value (setting during run)
        for col in PRESSURE_COLS + PROCESS_COLS:
            if col in group.columns:
                rec[f"{col}_mean"] = group[col].mean()

        records.append(rec)

    return pd.DataFrame(records)


def load_all_runs(split: str = "training") -> pd.DataFrame:
    """Load and aggregate all lot files into per-run rows."""
    data_dir = os.path.join(CMP1_DIR, "CMP-data", split)
    files = sorted(glob.glob(os.path.join(data_dir, f"CMP-{split}-*.csv")))
    logger.info(f"Loading {len(files)} {split} lot files...")

    all_runs = []
    for file_id, fpath in enumerate(files):
        try:
            runs = aggregate_runs_from_file(fpath, file_id)
            all_runs.append(runs)
        except Exception as e:
            logger.warning(f"Error in {fpath}: {e}")

    result = pd.concat(all_runs, ignore_index=True)
    logger.info(f"{len(result)} runs aggregated from {len(files)} lot files")
    return result


# ============================================================
# Step 2: Removal Rate Matching
# ============================================================

def merge_with_rr(runs_df: pd.DataFrame, split: str = "training") -> pd.DataFrame:
    """
    Join run-aggregated data with AVG_REMOVAL_RATE.
    Removes outliers (RR > 500).
    """
    rr_path = os.path.join(CMP1_DIR, f"CMP-{split}-removalrate.csv")
    rr_df = pd.read_csv(rr_path)

    # outlier removal
    n_before = len(rr_df)
    rr_df = rr_df[rr_df["AVG_REMOVAL_RATE"] <= RR_OUTLIER_THRESHOLD].copy()
    logger.info(f"RR outlier removal: {n_before} -> {len(rr_df)}")

    # join on WAFER_ID + STAGE
    merged = runs_df.merge(rr_df, on=["WAFER_ID", "STAGE"], how="inner")
    logger.info(f"After RR join: {len(merged)} runs (before: {len(runs_df)})")
    return merged


# ============================================================
# Step 3: Sort sequences and build MDP tuples
# ============================================================

def build_mdp_tuples(
    runs_df: pd.DataFrame,
    target_rr_dict: Dict[str, float],
    spec_margin_dict: Optional[Dict[str, float]] = None,
    cross_lot: bool = False,
    lambda_s: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Sort by time and generate (s, a, r, s', done) tuples + drift features.

    Unified reward: r = -alpha*(e/sigma_spec)^2 - lambda*||Delta_a||_1, clip[-10,0]

    Args:
        target_rr_dict: per-STAGE target RR {"A": 97.6, "B": 80.0}
        spec_margin_dict: per-STAGE spec margin (sigma_spec). If None, uses per-stage RR std.
        cross_lot: if True, ignore lot boundaries and concatenate all lots
                   within (STAGE, CHAMBER) into one long sequence.
    """
    wear_last_cols    = [f"{c}_last" for c in WEAR_COLS]
    pressure_mean_cols = [f"{c}_mean" for c in PRESSURE_COLS]
    process_mean_cols  = [f"{c}_mean" for c in PROCESS_COLS]

    observations      = []
    actions           = []
    rewards           = []
    next_observations = []
    terminals         = []
    drift_features    = []
    next_drift_features = []
    lot_file_ids      = []  # lot file index per transition (for chronological split)

    # group key selection: ignore lot boundaries if cross_lot
    group_keys = ["STAGE", "CHAMBER"] if cross_lot else ["file_id", "STAGE", "CHAMBER"]
    groups = runs_df.groupby(group_keys)
    n_seq = 0

    for group_key, group in groups:
        if cross_lot:
            stage, chamber = group_key
            sort_cols = ["file_id", "first_timestamp"]
        else:
            file_id, stage, chamber = group_key
            sort_cols = ["first_timestamp"]

        group = group.sort_values(sort_cols).reset_index(drop=True)

        if len(group) < 2:
            continue

        target_rr = target_rr_dict.get(stage, runs_df["AVG_REMOVAL_RATE"].median())
        lot_size  = len(group)

        for i in range(len(group) - 1):
            row_t   = group.iloc[i]
            row_tp1 = group.iloc[i + 1]

            # file_id: use run's own file_id in cross_lot mode
            cur_file_id = int(row_t["file_id"]) if cross_lot else int(file_id)

            # --- State: wear(6) + process(6) + prev_RR(1) = 13D ---
            wear_t = np.array([row_t.get(c, 0.0) for c in wear_last_cols], dtype=np.float32)
            proc_t = np.array([row_t.get(c, 0.0) for c in process_mean_cols], dtype=np.float32)
            rr_t   = np.array([row_t["AVG_REMOVAL_RATE"]], dtype=np.float32)
            s_t    = np.concatenate([wear_t, proc_t, rr_t])

            wear_tp1 = np.array([row_tp1.get(c, 0.0) for c in wear_last_cols], dtype=np.float32)
            proc_tp1 = np.array([row_tp1.get(c, 0.0) for c in process_mean_cols], dtype=np.float32)
            rr_tp1   = np.array([row_tp1["AVG_REMOVAL_RATE"]], dtype=np.float32)
            s_tp1    = np.concatenate([wear_tp1, proc_tp1, rr_tp1])

            # --- Action: pressure of run t+1 ---
            a_t = np.array([row_tp1.get(c, 0.0) for c in pressure_mean_cols], dtype=np.float32)

            # --- Reward: unified r = -alpha*(e/sigma)^2 - lambda*||Delta_a||_1 ---
            from src.rl.reward import compute_reward as _compute_reward
            actual_rr = float(row_tp1["AVG_REMOVAL_RATE"])
            spec_margin = spec_margin_dict.get(stage, 30.0) if spec_margin_dict else 30.0
            if i > 0:
                a_prev = np.array([row_t.get(c, 0.0) for c in pressure_mean_cols], dtype=np.float32)
            else:
                a_prev = None
            reward = _compute_reward(actual_rr, target_rr, a_t, a_prev,
                                    lambda_s=lambda_s, spec_margin=spec_margin)

            # --- Terminal ---
            terminal = 1.0 if i == len(group) - 2 else 0.0

            # --- Drift features: wear(6) + delta_wear(6) + position(1) = 13D ---
            delta_wear = wear_t - np.array(
                [group.iloc[i - 1].get(c, wear_t[j]) for j, c in enumerate(wear_last_cols)],
                dtype=np.float32
            ) if i > 0 else np.zeros(len(WEAR_COLS), dtype=np.float32)
            lot_pos = np.array([i / max(lot_size - 1, 1)], dtype=np.float32)
            d_t     = np.concatenate([wear_t, delta_wear, lot_pos])

            delta_wear_tp1 = wear_tp1 - wear_t
            lot_pos_tp1    = np.array([(i + 1) / max(lot_size - 1, 1)], dtype=np.float32)
            d_tp1          = np.concatenate([wear_tp1, delta_wear_tp1, lot_pos_tp1])

            observations.append(s_t)
            actions.append(a_t)
            rewards.append(reward)
            next_observations.append(s_tp1)
            terminals.append(terminal)
            drift_features.append(d_t)
            next_drift_features.append(d_tp1)
            lot_file_ids.append(cur_file_id)

        n_seq += 1

    logger.info(f"{n_seq} sequences -> {len(observations)} MDP tuples")

    # stage_labels: 0 = Stage A, 1 = Stage B
    stage_labels = []
    for group_key, group in runs_df.groupby(group_keys):
        if cross_lot:
            stage, chamber = group_key
            sort_cols = ["file_id", "first_timestamp"]
        else:
            file_id, stage, chamber = group_key
            sort_cols = ["first_timestamp"]
        group = group.sort_values(sort_cols).reset_index(drop=True)
        if len(group) < 2:
            continue
        label = 0 if stage == "A" else 1
        for i in range(len(group) - 1):
            stage_labels.append(label)

    result = {
        "observations":       np.array(observations,       dtype=np.float32),
        "actions":            np.array(actions,            dtype=np.float32),
        "rewards":            np.array(rewards,            dtype=np.float32),
        "next_observations":  np.array(next_observations,  dtype=np.float32),
        "terminals":          np.array(terminals,          dtype=np.float32),
        "drift_features":     np.array(drift_features,     dtype=np.float32),
        "next_drift_features": np.array(next_drift_features, dtype=np.float32),
        "stage_labels":       np.array(stage_labels,       dtype=np.int32),
        "file_ids":           np.array(lot_file_ids,       dtype=np.int32),
    }

    logger.info(
        f"State: {result['observations'].shape[1]}D, "
        f"Action: {result['actions'].shape[1]}D, "
        f"Drift: {result['drift_features'].shape[1]}D"
    )
    return result


# ============================================================
# Full Pipeline
# ============================================================

def preprocess_cmp1_r2r(
    config: Optional[DataConfig] = None,
    cross_lot: bool = False,
    lambda_s: float = 0.01,
) -> Tuple[Dict[str, np.ndarray], Dict, StandardScaler, StandardScaler]:
    """
    Full preprocessing pipeline for CMP1 R2R control.

    Returns:
        (mdp_data, target_rr_dict, state_scaler, action_scaler)
    """
    if config is None:
        config = DataConfig()

    # 1. Load & aggregate runs
    runs_df = load_all_runs("training")

    # 2. Merge with removal rates
    runs_df = merge_with_rr(runs_df, "training")

    # 3. Target RR and spec margin: per-STAGE median and std
    target_rr_dict = {}
    spec_margin_dict = {}
    for stage, grp in runs_df.groupby("STAGE"):
        target_rr_dict[stage] = float(grp["AVG_REMOVAL_RATE"].median())
        spec_margin_dict[stage] = float(grp["AVG_REMOVAL_RATE"].std())
        logger.info(
            f"Stage {stage}: target_RR={target_rr_dict[stage]:.2f}, "
            f"mean={grp['AVG_REMOVAL_RATE'].mean():.2f}, "
            f"std(=sigma_spec)={spec_margin_dict[stage]:.2f}, "
            f"n={len(grp)}"
        )

    # 4. Build MDP tuples
    mdp_data = build_mdp_tuples(runs_df, target_rr_dict, spec_margin_dict=spec_margin_dict,
                                cross_lot=cross_lot, lambda_s=lambda_s)

    # 5. Normalize (state & action)
    state_scaler  = StandardScaler()
    action_scaler = StandardScaler()

    if config.normalize:
        mdp_data["observations"]      = state_scaler.fit_transform(mdp_data["observations"])
        mdp_data["next_observations"] = state_scaler.transform(mdp_data["next_observations"])
        mdp_data["actions"]           = action_scaler.fit_transform(mdp_data["actions"])

    logger.info(
        f"CMP1 R2R preprocessing done: {len(mdp_data['observations'])} transitions, "
        f"reward range [{mdp_data['rewards'].min():.4f}, {mdp_data['rewards'].max():.4f}]"
    )
    return mdp_data, target_rr_dict, state_scaler, action_scaler


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mdp_data, targets, s_scaler, a_scaler = preprocess_cmp1_r2r()

    print("\n" + "=" * 60)
    print("CMP1 R2R Preprocessing Summary")
    print("=" * 60)
    print(f"Total MDP tuples: {len(mdp_data['observations'])}")
    print(f"State dim: {mdp_data['observations'].shape[1]}D")
    print(f"Action dim: {mdp_data['actions'].shape[1]}D")
    print(f"Drift dim: {mdp_data['drift_features'].shape[1]}D")
    print(f"Reward range: [{mdp_data['rewards'].min():.4f}, {mdp_data['rewards'].max():.4f}]")
    print(f"Terminal ratio: {mdp_data['terminals'].mean():.4f}")
    print("\nTarget RR (per stage):")
    for stage, target in targets.items():
        print(f"  Stage {stage}: {target:.2f}")
