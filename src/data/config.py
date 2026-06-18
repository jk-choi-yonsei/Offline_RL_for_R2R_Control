"""
Configuration for Offline RL R2R Control.
Defines RL variable mappings, hyperparameters, and data paths.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================
# Paths
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(_PROJECT_ROOT, "Dataset")
CMP1_DIR = os.path.join(DATASET_DIR, "CMP1")
CMP2_DIR = os.path.join(DATASET_DIR, "CMP2")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")


# ============================================================
# CMP2 Variable Mapping (Primary Dataset)
# ============================================================

# --- Removal Rate (target property) ---
RR_COLUMNS = ["AKE3", "AK", "AKE1", "AKE2"]          # A-side removal rates
RR_COLUMNS_B = ["BK", "BKE1", "BKE2", "BKE3"]        # B-side (often #N/A)

# --- State Variables (observable, non-controllable) ---
STATE_WEAR_COLUMNS = ["D", "M", "P", "R"]              # consumable wear (drift)
STATE_TEMP_COLUMNS = ["T1_04", "T2_04", "T3_04", "T3_05", "T3_06"]  # temperatures

# --- Action Variables (controllable) ---
ACTION_COLUMNS = ["P3_04_Z1", "P3_04_Z3", "P3_04_Z5", "P3_04_Z6", "P3_04_Z7"]  # Zone Pressure

# --- Metadata (for filtering/grouping) ---
META_COLUMNS = ["WF_ID", "MEQ", "PREEQ", "LOT_ID", "TAG", "sequence"]

# Combined state: wear + temperature + previous RR
STATE_COLUMNS = STATE_WEAR_COLUMNS + STATE_TEMP_COLUMNS  # 9 base dims
# adding the previous run's RR to the state adds +4 dims (AKE3_prev, AK_prev, AKE1_prev, AKE2_prev) = 13 dims


# ============================================================
# CMP1 Variable Mapping (Secondary Dataset)
# ============================================================

CMP1_META_COLUMNS = ["MACHINE_ID", "MACHINE_DATA", "TIMESTAMP", "WAFER_ID", "STAGE", "CHAMBER"]

CMP1_WEAR_COLUMNS = [
    "USAGE_OF_BACKING_FILM", "USAGE_OF_DRESSER",
    "USAGE_OF_POLISHING_TABLE", "USAGE_OF_DRESSER_TABLE",
    "USAGE_OF_MEMBRANE", "USAGE_OF_PRESSURIZED_SHEET",
]

CMP1_PRESSURE_COLUMNS = [
    "PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE",
    "CENTER_AIR_BAG_PRESSURE", "RETAINER_RING_PRESSURE",
    "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE",
]

CMP1_PROCESS_COLUMNS = [
    "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION",
    "DRESSING_WATER_STATUS",
]


# ============================================================
# Reward Function Parameters
# ============================================================

@dataclass
class RewardConfig:
    """Reward function configuration."""
    # Target RR: to be estimated from data statistics
    target_rr: Optional[float] = None  # Set after data analysis

    # Reward weights
    alpha: float = 1.0       # removal-rate accuracy weight
    beta: float = 0.01       # control-cost weight
    gamma: float = 5.0       # spec-violation penalty weight

    # Spec margin (tolerance)
    # AKE3 target 2850 +/- 450 -> spec_margin = 450
    spec_margin: float = 450.0  # in RR units

    # per-zone-pressure cost coefficients (Z1, Z3, Z5, Z6, Z7)
    cost_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])


# ============================================================
# World Model Parameters
# ============================================================

@dataclass
class WorldModelConfig:
    """World Model (Ensemble Dynamics) configuration."""
    ensemble_size: int = 5
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20      # Early stopping patience
    dropout: float = 0.1
    uncertainty_penalty: float = 1.0  # λ for MOPO-style penalty


# ============================================================
# Offline RL Parameters
# ============================================================

@dataclass
class OfflineRLConfig:
    """Offline RL training configuration."""
    algorithm: str = "cql"
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 100
    gamma: float = 0.99      # Discount factor
    tau: float = 0.005        # Target network soft update

    # CQL specific
    cql_alpha: float = 1.0


# ============================================================
# Data Split Configuration
# ============================================================

@dataclass
class DataConfig:
    """Data processing configuration."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    normalize: bool = True
    include_prev_rr_in_state: bool = True  # include the previous run's RR in the state
    chambers: List[str] = field(default_factory=lambda: ["C1", "C3", "C6"])
