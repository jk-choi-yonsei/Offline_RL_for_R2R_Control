"""
Ensemble MLP Dynamics Model for World Model.
Predicts next state and reward given current state and action.
Uses ensemble of MLPs for uncertainty quantification.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DynamicsMLP(nn.Module):
    """
    Single MLP dynamics model.
    Predicts Gaussian distribution over next state and reward.
    Output: (mean, log_var) for each output dimension.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        dropout: float = 0.1,
        predict_reward: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.predict_reward = predict_reward
        self.output_dim = state_dim + (1 if predict_reward else 0)

        # Build network
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, self.output_dim)
        self.log_var_head = nn.Linear(input_dim, self.output_dim)

        # Bound log variance to prevent numerical issues
        self.min_log_var = nn.Parameter(-10.0 * torch.ones(self.output_dim), requires_grad=True)
        self.max_log_var = nn.Parameter(0.5 * torch.ones(self.output_dim), requires_grad=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=1 / (2 * m.in_features ** 0.5))
                nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)

        Returns:
            mean: (batch_size, output_dim) - predicted mean
            log_var: (batch_size, output_dim) - predicted log variance
        """
        x = torch.cat([state, action], dim=-1)
        h = self.backbone(x)

        mean = self.mean_head(h)
        log_var = self.log_var_head(h)

        # Soft clamp log variance
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)

        return mean, log_var

    def get_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood loss.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            target: (batch_size, output_dim) - ground truth [next_state, reward]

        Returns:
            Scalar loss.
        """
        mean, log_var = self.forward(state, action)
        var = torch.exp(log_var)

        # Gaussian NLL
        nll = 0.5 * (log_var + (target - mean) ** 2 / var)
        loss = nll.mean()

        # Regularize log variance bounds
        loss += 0.01 * (self.max_log_var.sum() - self.min_log_var.sum())

        return loss


class EnsembleDynamics(nn.Module):
    """
    Ensemble of MLP dynamics models.
    Provides uncertainty estimation via prediction disagreement.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dims: List[int] = [256, 256, 256],
        dropout: float = 0.1,
        predict_reward: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.predict_reward = predict_reward

        # Create ensemble
        self.models = nn.ModuleList([
            DynamicsMLP(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                predict_reward=predict_reward,
            )
            for _ in range(ensemble_size)
        ])

        logger.info(
            f"EnsembleDynamics: {ensemble_size} models, "
            f"state_dim={state_dim}, action_dim={action_dim}"
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all ensemble members.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)

        Returns:
            ensemble_means: (ensemble_size, batch_size, output_dim)
            ensemble_vars: (ensemble_size, batch_size, output_dim)
            uncertainty: (batch_size,) - ensemble disagreement
        """
        means = []
        log_vars = []

        for model in self.models:
            mean, log_var = model(state, action)
            means.append(mean)
            log_vars.append(log_var)

        ensemble_means = torch.stack(means)      # (K, B, D)
        ensemble_vars = torch.exp(torch.stack(log_vars))  # (K, B, D)

        # Uncertainty: standard deviation of means across ensemble members
        # (epistemic uncertainty)
        uncertainty = ensemble_means.std(dim=0).mean(dim=-1)  # (B,)

        return ensemble_means, ensemble_vars, uncertainty

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward with uncertainty.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            deterministic: If True, use mean prediction; else sample.

        Returns:
            next_state: (batch_size, state_dim)
            reward: (batch_size, 1) or None
            uncertainty: (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            ensemble_means, ensemble_vars, uncertainty = self.forward(state, action)

            # Average prediction across ensemble
            mean = ensemble_means.mean(dim=0)  # (B, D)

            if deterministic:
                prediction = mean
            else:
                # Sample from mixture of Gaussians (approx. by avg then sample)
                avg_var = ensemble_vars.mean(dim=0) + ensemble_means.var(dim=0)
                prediction = mean + torch.randn_like(mean) * avg_var.sqrt()

            # Split into next_state and reward
            next_state = prediction[:, :self.state_dim]
            reward = prediction[:, self.state_dim:] if self.predict_reward else None

        return next_state, reward, uncertainty

    def get_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        model_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute loss for one or all ensemble members.

        Args:
            state, action, target: Training data.
            model_idx: If None, compute loss for all models.

        Returns:
            Loss tensor.
        """
        if model_idx is not None:
            return self.models[model_idx].get_loss(state, action, target)

        total_loss = torch.tensor(0.0, device=state.device)
        for model in self.models:
            total_loss += model.get_loss(state, action, target)
        return total_loss / self.ensemble_size
