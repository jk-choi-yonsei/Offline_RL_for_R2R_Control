"""
World Model wrapper.
Manages ensemble dynamics, training, rollouts, and uncertainty-penalized rewards.
"""

import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.data.config import WorldModelConfig, CHECKPOINT_DIR
from src.models.dynamics_mlp import EnsembleDynamics

logger = logging.getLogger(__name__)


class WorldModel:
    """
    World Model for Model-based Offline RL.
    Wraps EnsembleDynamics with training, inference, and rollout capabilities.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[WorldModelConfig] = None,
        device: str = "auto",
    ):
        if config is None:
            config = WorldModelConfig()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Device
        if device == "auto":
            self.device = torch.device("mps" if torch.backends.mps.is_available()
                                       else "cuda" if torch.cuda.is_available()
                                       else "cpu")
        else:
            self.device = torch.device(device)

        # Build ensemble
        self.ensemble = EnsembleDynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            ensemble_size=config.ensemble_size,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            predict_reward=True,
        ).to(self.device)

        # Optimizers (one per ensemble member for bootstrap)
        self.optimizers = [
            optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            for model in self.ensemble.models
        ]

        self.trained = False
        logger.info(f"WorldModel initialized on {self.device}")

    def train(
        self,
        mdp_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Train the ensemble dynamics model.

        Each member is trained on a bootstrap sample of the data.

        Args:
            mdp_data: Training MDP data.
            val_data: Optional validation MDP data.
            log_dir: TensorBoard log directory.

        Returns:
            Training history dictionary.
        """
        config = self.config
        writer = SummaryWriter(log_dir) if log_dir else None

        # Prepare training targets: [delta_state, reward]
        # Using delta (next_state - state) instead of absolute next_state so the
        # network focuses on learning the action-induced *change*, isolating the
        # weak action signal (corr ~0.10) from the dominant state-persistence term.
        states = torch.FloatTensor(mdp_data["observations"])
        actions = torch.FloatTensor(mdp_data["actions"])
        next_states = torch.FloatTensor(mdp_data["next_observations"])
        rewards = torch.FloatTensor(mdp_data["rewards"]).unsqueeze(-1)
        delta_states = next_states - states
        targets = torch.cat([delta_states, rewards], dim=-1)

        n_samples = len(states)
        history = {"train_loss": [], "val_loss": []}

        # Validation data
        if val_data is not None:
            val_states = torch.FloatTensor(val_data["observations"]).to(self.device)
            val_actions = torch.FloatTensor(val_data["actions"]).to(self.device)
            val_next = torch.FloatTensor(val_data["next_observations"]).to(self.device)
            val_rewards = torch.FloatTensor(val_data["rewards"]).unsqueeze(-1).to(self.device)
            val_delta = val_next - val_states
            val_targets = torch.cat([val_delta, val_rewards], dim=-1)

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training World Model: {n_samples} samples, {config.max_epochs} max epochs")

        for epoch in range(config.max_epochs):
            self.ensemble.train()
            epoch_losses = []

            for model_idx in range(config.ensemble_size):
                # Bootstrap sampling for each ensemble member
                bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
                bs_states = states[bootstrap_idx].to(self.device)
                bs_actions = actions[bootstrap_idx].to(self.device)
                bs_targets = targets[bootstrap_idx].to(self.device)

                # Mini-batch training
                dataset = TensorDataset(bs_states, bs_actions, bs_targets)
                loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

                model_loss = 0.0
                n_batches = 0
                for batch_s, batch_a, batch_t in loader:
                    self.optimizers[model_idx].zero_grad()
                    loss = self.ensemble.get_loss(batch_s, batch_a, batch_t, model_idx)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ensemble.models[model_idx].parameters(), 1.0)
                    self.optimizers[model_idx].step()
                    model_loss += loss.item()
                    n_batches += 1

                epoch_losses.append(model_loss / max(n_batches, 1))

            avg_train_loss = np.mean(epoch_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if val_data is not None:
                self.ensemble.eval()
                with torch.no_grad():
                    val_loss = self.ensemble.get_loss(val_states, val_actions, val_targets)
                    val_loss_val = val_loss.item()
                history["val_loss"].append(val_loss_val)

                # Early stopping
                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    patience_counter += 1
            else:
                val_loss_val = avg_train_loss

            # Logging
            if writer:
                writer.add_scalar("world_model/train_loss", avg_train_loss, epoch)
                if val_data is not None:
                    writer.add_scalar("world_model/val_loss", val_loss_val, epoch)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{config.max_epochs} | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {val_loss_val:.6f} | "
                    f"Patience: {patience_counter}/{config.patience}"
                )

            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if val_data is not None and os.path.exists(
            os.path.join(CHECKPOINT_DIR, "world_model_best.pt")
        ):
            self._load_checkpoint("best")

        self.trained = True
        if writer:
            writer.close()

        logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
        return history

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict next state and reward with uncertainty.

        Returns:
            next_state, reward, uncertainty (all numpy arrays)
        """
        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.FloatTensor(action).to(self.device)

        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            action_t = action_t.unsqueeze(0)

        delta_state, reward, uncertainty = self.ensemble.predict(
            state_t, action_t, deterministic=deterministic
        )
        next_state = delta_state + state_t  # delta prediction → absolute next state

        return (
            next_state.cpu().numpy(),
            reward.cpu().numpy() if reward is not None else None,
            uncertainty.cpu().numpy(),
        )

    def rollout(
        self,
        initial_state: np.ndarray,
        policy,
        horizon: int = 5,
        penalty_coeff: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate virtual rollout using World Model.

        Args:
            initial_state: Starting state(s).
            policy: Callable that maps state → action.
            horizon: Number of steps to roll out.
            penalty_coeff: Uncertainty penalty coefficient (MOPO λ).

        Returns:
            Dictionary with rollout trajectories.
        """
        if penalty_coeff is None:
            penalty_coeff = self.config.uncertainty_penalty

        states = [initial_state]
        actions_list = []
        rewards_list = []
        uncertainties = []

        current_state = initial_state
        for t in range(horizon):
            # Get action from policy
            action = policy(current_state)

            # Predict next state and reward
            next_state, reward, uncertainty = self.predict(
                current_state, action, deterministic=False
            )

            # Apply uncertainty penalty (MOPO)
            penalized_reward = reward - penalty_coeff * uncertainty.reshape(-1, 1)

            states.append(next_state)
            actions_list.append(action)
            rewards_list.append(penalized_reward)
            uncertainties.append(uncertainty)

            current_state = next_state

        return {
            "observations": np.concatenate(states[:-1]),
            "actions": np.concatenate(actions_list),
            "rewards": np.concatenate(rewards_list).squeeze(),
            "next_observations": np.concatenate(states[1:]),
            "terminals": np.zeros(len(actions_list) * initial_state.shape[0]),
            "uncertainties": np.concatenate(uncertainties),
        }

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f"world_model_{name}.pt")
        torch.save({
            "ensemble_state_dict": self.ensemble.state_dict(),
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)

    def _load_checkpoint(self, name: str):
        """Load model checkpoint."""
        path = os.path.join(CHECKPOINT_DIR, f"world_model_{name}.pt")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.ensemble.load_state_dict(checkpoint["ensemble_state_dict"])
        logger.info(f"Loaded checkpoint: {path}")
