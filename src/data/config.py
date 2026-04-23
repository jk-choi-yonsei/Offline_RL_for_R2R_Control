"""
Configuration for Offline RL R2R Control.
Defines hyperparameters and data paths.
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
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")


# ============================================================
# CMP1 Variable Mapping (Public PHM Dataset)
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
    target_rr: Optional[float] = None

    # Reward weights (paper: alpha=1.0, lambda_s=0.01)
    alpha: float = 1.0
    beta: float = 0.01
    gamma: float = 5.0

    spec_margin: float = 450.0  # RR spec half-width (Angstrom)
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
    patience: int = 20
    dropout: float = 0.1
    uncertainty_penalty: float = 1.0


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
    gamma: float = 0.99
    tau: float = 0.005
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
    include_prev_rr_in_state: bool = True
