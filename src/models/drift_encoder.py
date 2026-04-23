"""
Drift Encoder for SARC.
Encodes equipment wear (drift) features into a compact context vector.
Supports pre-training via RR-prediction auxiliary task.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DriftEncoder(nn.Module):
    """
    Encodes drift (wear) features → compact context vector.

    Input: [D, M, P, R, ΔD, ΔM, ΔP, ΔR, lot_position] = 9D
    Output: context vector (context_dim)

    Can be pre-trained via auxiliary task: drift → RR prediction.
    """

    def __init__(
        self,
        drift_dim: int = 9,
        context_dim: int = 2,
        hidden_dim: int = 64,
        rr_out_dim: int = 4,
    ):
        super().__init__()
        self.drift_dim = drift_dim
        self.context_dim = context_dim
        self.rr_out_dim = rr_out_dim

        # Encoder: drift features → context
        self.encoder = nn.Sequential(
            nn.Linear(drift_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.Tanh(),  # bounded context for stable RL
        )

        # Auxiliary head for pre-training: context → RR prediction
        self.aux_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rr_out_dim),
        )

        self._init_weights()
        logger.info(f"DriftEncoder: {drift_dim}D → {context_dim}D context")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drift_features: torch.Tensor) -> torch.Tensor:
        """
        Encode drift features to context vector.

        Args:
            drift_features: (batch, drift_dim)

        Returns:
            context: (batch, context_dim)
        """
        return self.encoder(drift_features)

    def predict_rr(self, drift_features: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary task: predict RR from drift features (for pre-training).

        Returns:
            predicted_rr: (batch, rr_out_dim)
        """
        context = self.forward(drift_features)
        return self.aux_head(context)

    def pretrain_loss(
        self,
        drift_features: torch.Tensor,
        target_rr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pre-training loss: MSE(predicted_rr, actual_rr).

        Args:
            drift_features: (batch, drift_dim)
            target_rr: (batch, 4) — actual RR values

        Returns:
            Scalar loss.
        """
        pred_rr = self.predict_rr(drift_features)
        return F.mse_loss(pred_rr, target_rr)
