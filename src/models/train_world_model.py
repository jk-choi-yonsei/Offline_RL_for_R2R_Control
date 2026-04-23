"""
World Model training script.
Train ensemble dynamics model on preprocessed CMP2 data.
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.config import WorldModelConfig, DataConfig, RESULTS_DIR
from src.data.preprocess_cmp2 import preprocess_cmp2
from src.data.mdp_dataset import split_data, save_dataset
from src.models.world_model import WorldModel


def main():
    parser = argparse.ArgumentParser(description="Train World Model")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- 1. Data Preprocessing ---
    logger.info("=" * 60)
    logger.info("Step 1: Data Preprocessing")
    logger.info("=" * 60)

    mdp_data, targets, s_scaler, a_scaler = preprocess_cmp2()
    train_data, val_data, test_data = split_data(mdp_data)

    # Save processed data
    save_dataset(
        train_data,
        scalers={"state": s_scaler, "action": a_scaler},
        target_rr=targets,
        name="cmp2_train",
    )
    save_dataset(val_data, scalers={}, target_rr=targets, name="cmp2_val")
    save_dataset(test_data, scalers={}, target_rr=targets, name="cmp2_test")

    state_dim = train_data["observations"].shape[1]
    action_dim = train_data["actions"].shape[1]

    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    logger.info(f"Train: {len(train_data['observations'])}, "
                f"Val: {len(val_data['observations'])}, "
                f"Test: {len(test_data['observations'])}")

    # --- 2. World Model Training ---
    logger.info("=" * 60)
    logger.info("Step 2: World Model Training")
    logger.info("=" * 60)

    wm_config = WorldModelConfig(
        ensemble_size=args.ensemble_size,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )

    world_model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        config=wm_config,
        device=args.device,
    )

    log_dir = args.log_dir or os.path.join(RESULTS_DIR, "logs", "world_model")
    history = world_model.train(
        mdp_data=train_data,
        val_data=val_data,
        log_dir=log_dir,
    )

    # --- 3. Evaluation ---
    logger.info("=" * 60)
    logger.info("Step 3: Evaluation")
    logger.info("=" * 60)

    import numpy as np

    # Evaluate on test set
    test_states = test_data["observations"]
    test_actions = test_data["actions"]
    test_next_states = test_data["next_observations"]
    test_rewards = test_data["rewards"]

    pred_next, pred_reward, uncertainties = world_model.predict(
        test_states, test_actions, deterministic=True
    )

    # Next state prediction error
    state_mse = np.mean((pred_next - test_next_states) ** 2)
    state_mae = np.mean(np.abs(pred_next - test_next_states))

    # Reward prediction error
    if pred_reward is not None:
        reward_mse = np.mean((pred_reward.squeeze() - test_rewards) ** 2)
        reward_mae = np.mean(np.abs(pred_reward.squeeze() - test_rewards))
    else:
        reward_mse = reward_mae = float("nan")

    # Uncertainty statistics
    unc_mean = np.mean(uncertainties)
    unc_std = np.std(uncertainties)

    logger.info(f"Test Results:")
    logger.info(f"  State MSE: {state_mse:.6f}, MAE: {state_mae:.6f}")
    logger.info(f"  Reward MSE: {reward_mse:.6f}, MAE: {reward_mae:.6f}")
    logger.info(f"  Uncertainty: mean={unc_mean:.6f}, std={unc_std:.6f}")

    # Save final model
    world_model._save_checkpoint("final")
    logger.info("World Model training complete!")


if __name__ == "__main__":
    main()
