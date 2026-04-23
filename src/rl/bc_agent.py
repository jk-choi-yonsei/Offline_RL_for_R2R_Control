"""
Behavioral Cloning (BC) Agent for R2R Control.

Simplest offline baseline: supervised learning from (state, drift) -> action.
Uses the same SARCActor + DriftEncoder architecture as SARC for fair comparison.
No critic, no RL objective — pure MSE imitation of behavioral data.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.sarc_agent import SARCActor

logger = logging.getLogger(__name__)


class BCAgent:
    """
    Behavioral Cloning agent.

    Learns π(s, c) → a by minimizing MSE(π(s,c), a_behavioral).
    DriftEncoder optionally pre-trained via auxiliary RR prediction task,
    identical to SARC setup for fair ablation.

    use_drift=False: state-only policy (no drift encoder).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        drift_dim: int = 9,
        context_dim: int = 2,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        lr_encoder: float = 1e-3,
        rr_out_dim: int = 4,
        use_drift: bool = True,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device(
                "mps"  if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.use_drift   = use_drift
        self.context_dim = context_dim if use_drift else 0

        from src.models.drift_encoder import DriftEncoder

        if use_drift:
            self.drift_encoder = DriftEncoder(
                drift_dim=drift_dim,
                context_dim=context_dim,
                hidden_dim=64,
                rr_out_dim=rr_out_dim,
            ).to(self.device)
            self.encoder_optim = torch.optim.Adam(
                self.drift_encoder.parameters(), lr=lr_encoder
            )
        else:
            self.drift_encoder = None
            self.encoder_optim = None

        self.actor = SARCActor(
            state_dim, action_dim, self.context_dim, hidden_dim
        ).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)

        logger.info(
            f"BCAgent: state={state_dim}, action={action_dim}, "
            f"context={self.context_dim}, use_drift={use_drift}, device={self.device}"
        )

    def _encode(self, drift: torch.Tensor) -> torch.Tensor:
        if self.use_drift:
            return self.drift_encoder(drift)
        return torch.zeros(drift.size(0), 0, device=self.device)

    def pretrain_encoder(
        self,
        drift_features: torch.Tensor,
        target_rr: torch.Tensor,
        n_epochs: int = 50,
        batch_size: int = 256,
    ) -> list:
        """Pre-train DriftEncoder via auxiliary RR prediction (same as SARC)."""
        if not self.use_drift:
            return [0.0]

        self.drift_encoder.train()
        losses = []
        for epoch in range(n_epochs):
            idx = torch.randperm(len(drift_features))
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, len(drift_features), batch_size):
                b = idx[start:start + batch_size]
                d  = drift_features[b].to(self.device)
                rr = target_rr[b].to(self.device)
                loss = self.drift_encoder.pretrain_loss(d, rr)
                self.encoder_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                epoch_loss += loss.item()
                n_batches  += 1
            losses.append(epoch_loss / max(n_batches, 1))
            if (epoch + 1) % 10 == 0:
                logger.info(f"BC encoder pretrain {epoch+1}/{n_epochs} | "
                            f"loss={losses[-1]:.6f}")
        return losses

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        drift: torch.Tensor,
        max_grad_norm: float = 1.0,
    ) -> float:
        """Single BC update: MSE(predicted_action, behavioral_action)."""
        self.actor.train()
        if self.drift_encoder is not None:
            self.drift_encoder.train()

        context    = self._encode(drift)
        pred_actions = self.actor(states, context)
        loss = F.mse_loss(pred_actions, actions)

        self.actor_optim.zero_grad()
        if self.encoder_optim is not None:
            self.encoder_optim.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        if self.drift_encoder is not None:
            nn.utils.clip_grad_norm_(self.drift_encoder.parameters(), max_grad_norm)

        self.actor_optim.step()
        if self.encoder_optim is not None:
            self.encoder_optim.step()

        return loss.item()

    def select_action(self, state: np.ndarray, drift_features: np.ndarray) -> np.ndarray:
        self.actor.eval()
        if self.drift_encoder is not None:
            self.drift_encoder.eval()
        with torch.no_grad():
            s   = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            d   = torch.FloatTensor(drift_features).unsqueeze(0).to(self.device)
            ctx = self._encode(d)
            a   = self.actor(s, ctx)
        return a.cpu().numpy().squeeze(0)

    def save(self, path: str):
        ckpt = {"actor": self.actor.state_dict()}
        if self.drift_encoder is not None:
            ckpt["drift_encoder"] = self.drift_encoder.state_dict()
        torch.save(ckpt, path)
        logger.info(f"BC model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        if self.drift_encoder is not None and "drift_encoder" in ckpt:
            self.drift_encoder.load_state_dict(ckpt["drift_encoder"])
        logger.info(f"BC model loaded from {path}")
