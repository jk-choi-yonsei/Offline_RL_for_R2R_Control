"""
Train World Model on CMP1 MDP data (corrected R2R formulation).
Saves to results/checkpoints/world_model_cmp1_best.pt
"""

import logging
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.config import RESULTS_DIR, CHECKPOINT_DIR, WorldModelConfig
from src.models.world_model import WorldModel

logger = logging.getLogger(__name__)


def load_cmp1_npy(split: str) -> dict:
    data_dir = os.path.join(RESULTS_DIR, "data")
    keys = ["observations", "actions", "rewards", "next_observations", "terminals"]
    data = {}
    for k in keys:
        path = os.path.join(data_dir, f"cmp1_{split}_{k}.npy")
        data[k] = np.load(path)
    return data


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train_data = load_cmp1_npy("train")
    val_data   = load_cmp1_npy("val")
    test_data  = load_cmp1_npy("test")

    state_dim  = train_data["observations"].shape[1]   # 13
    action_dim = train_data["actions"].shape[1]        # 6

    logger.info(f"CMP1 WM — State: {state_dim}D, Action: {action_dim}D")
    logger.info(f"Train: {len(train_data['observations'])}, "
                f"Val: {len(val_data['observations'])}, "
                f"Test: {len(test_data['observations'])}")

    wm_config = WorldModelConfig(
        ensemble_size=5,
        max_epochs=300,
        batch_size=128,
        learning_rate=1e-3,
        patience=30,
    )

    cmp1_ckpt = os.path.join(CHECKPOINT_DIR, "world_model_cmp1_best.pt")
    cmp2_ckpt = os.path.join(CHECKPOINT_DIR, "world_model_best.pt")
    cmp2_backup = os.path.join(CHECKPOINT_DIR, "world_model_cmp2_best.pt")

    # Back up CMP2 WM so training doesn't overwrite it
    if os.path.exists(cmp2_ckpt):
        shutil.copy2(cmp2_ckpt, cmp2_backup)
        logger.info(f"CMP2 WM backed up to {cmp2_backup}")

    world_model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        config=wm_config,
        device="auto",
    )

    log_dir = os.path.join(RESULTS_DIR, "logs", "world_model_cmp1")
    world_model.train(train_data, val_data=val_data, log_dir=log_dir)

    # world_model_best.pt now holds the CMP1 WM — rename to cmp1_best
    if os.path.exists(cmp2_ckpt):
        shutil.copy2(cmp2_ckpt, cmp1_ckpt)
        logger.info(f"CMP1 WM saved to {cmp1_ckpt}")

    # Restore CMP2 WM
    if os.path.exists(cmp2_backup):
        shutil.copy2(cmp2_backup, cmp2_ckpt)
        logger.info(f"CMP2 WM restored from backup")

    # Load CMP1 WM for evaluation
    world_model._load_checkpoint("cmp1_best")

    # Evaluate
    pred_next, pred_reward, uncertainties = world_model.predict(
        test_data["observations"], test_data["actions"], deterministic=True
    )
    state_mae = np.mean(np.abs(pred_next - test_data["next_observations"]))
    rr_mae    = np.mean(np.abs(pred_next[:, -1] - test_data["next_observations"][:, -1]))
    logger.info(f"Test State MAE: {state_mae:.6f} | RR dim MAE: {rr_mae:.6f}")
    logger.info(f"Uncertainty: mean={np.mean(uncertainties):.4f}, std={np.std(uncertainties):.4f}")
    logger.info(f"CMP1 World Model saved to {cmp1_ckpt}")


if __name__ == "__main__":
    main()
