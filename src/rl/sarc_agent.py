"""
SARC Agent: Drift-Conditioned CQL for R2R Control.
Policy and Q-function are conditioned on drift context from DriftEncoder.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SARCActor(nn.Module):
    """
    Drift-conditioned deterministic policy.
    π(s_t, c_t) → a_t where c_t is drift context.
    """

    def __init__(self, state_dim: int, action_dim: int, context_dim: int = 8,
                 hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + context_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            context: (B, context_dim)
        Returns:
            action: (B, action_dim) in [-1, 1]
        """
        x = torch.cat([state, context], dim=-1)
        return self.net(x)


class SARCCritic(nn.Module):
    """
    Drift-conditioned twin Q-networks.
    Q(s_t, a_t, c_t) conditioned on drift context.
    """

    def __init__(self, state_dim: int, action_dim: int, context_dim: int = 8,
                 hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + action_dim + context_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action, context], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, state: torch.Tensor, action: torch.Tensor,
              context: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action, context)
        return torch.min(q1, q2)


class SARCAgent:
    """
    SARC: State-Adaptive R2R Control agent.

    Components:
      1. DriftEncoder: wear features → context vector
      2. SARCActor: drift-conditioned policy (CQL-trained)
      3. SARCCritic: drift-conditioned Q (with CQL regularization)

    use_drift=False: ablation variant — actor conditions only on state, no drift context.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        drift_dim: int = 9,
        context_dim: int = 2,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_encoder: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        cql_alpha: float = 1.0,
        cql_n_random: int = 10,
        rr_out_dim: int = 4,
        bc_weight: float = 0.0,
        use_drift: bool = True,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_drift = use_drift
        self.context_dim = context_dim if use_drift else 0
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.cql_n_random = cql_n_random
        self.bc_weight = bc_weight

        # Networks
        from src.models.drift_encoder import DriftEncoder

        if use_drift:
            self.drift_encoder = DriftEncoder(
                drift_dim=drift_dim,
                context_dim=context_dim,
                hidden_dim=64,
                rr_out_dim=rr_out_dim,
            ).to(self.device)
            self.encoder_optim = torch.optim.Adam(self.drift_encoder.parameters(), lr=lr_encoder)
        else:
            self.drift_encoder = None
            self.encoder_optim = None

        self.actor = SARCActor(state_dim, action_dim, self.context_dim, hidden_dim).to(self.device)

        self.critic = SARCCritic(state_dim, action_dim, self.context_dim, hidden_dim).to(self.device)
        self.target_critic = SARCCritic(state_dim, action_dim, self.context_dim, hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        logger.info(f"SARCAgent: state={state_dim}, action={action_dim}, "
                     f"drift={drift_dim}, context={self.context_dim}, "
                     f"use_drift={use_drift}, device={self.device}")

    def _encode(self, drift: torch.Tensor) -> torch.Tensor:
        """Encode drift features. Returns empty tensor if use_drift=False."""
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
        """
        Pre-train DriftEncoder via auxiliary RR prediction task.

        Args:
            drift_features: (N, drift_dim)
            target_rr: (N, 4) actual RR values

        Returns:
            List of pre-training losses.
        """
        if not self.use_drift:
            logger.info("use_drift=False: skipping encoder pretrain")
            return [0.0]

        self.drift_encoder.train()
        losses = []

        for epoch in range(n_epochs):
            idx = torch.randperm(len(drift_features))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(drift_features), batch_size):
                batch_idx = idx[start:start + batch_size]
                d = drift_features[batch_idx].to(self.device)
                rr = target_rr[batch_idx].to(self.device)

                loss = self.drift_encoder.pretrain_loss(d, rr)
                self.encoder_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Encoder pretrain epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}")

        return losses

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminals: torch.Tensor,
        drift: torch.Tensor,
        next_drift: torch.Tensor,
        max_grad_norm: float = 1.0,
    ) -> dict:
        """
        Single SARC update step (CQL + drift-conditioning).

        Returns:
            Dict of loss values.
        """
        # --- Encode drift context ---
        context = self._encode(drift)
        next_context = self._encode(next_drift)

        # --- Critic update ---
        with torch.no_grad():
            next_a = self.actor(next_states, next_context)
            target_q = rewards + self.gamma * (1 - terminals) * \
                       self.target_critic.q_min(next_states, next_a, next_context)

        q1, q2 = self.critic(states, actions, context)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # CQL regularization: penalize overestimation of OOD actions
        batch_size = states.size(0)
        random_actions = torch.FloatTensor(
            batch_size * self.cql_n_random, self.action_dim
        ).uniform_(-1, 1).to(self.device)

        # Repeat states and context for random action evaluation
        s_rep = states.unsqueeze(1).repeat(1, self.cql_n_random, 1).view(-1, self.state_dim)
        if self.context_dim > 0:
            c_rep = context.unsqueeze(1).repeat(1, self.cql_n_random, 1).view(-1, self.context_dim)
        else:
            c_rep = torch.zeros(batch_size * self.cql_n_random, 0, device=self.device)

        q1_rand, q2_rand = self.critic(s_rep, random_actions, c_rep)
        q1_rand = q1_rand.view(batch_size, self.cql_n_random, 1)
        q2_rand = q2_rand.view(batch_size, self.cql_n_random, 1)

        # CQL loss: logsumexp(Q_rand) - Q(s,a)
        cql_loss = (
            torch.logsumexp(q1_rand, dim=1).mean() - q1.mean() +
            torch.logsumexp(q2_rand, dim=1).mean() - q2.mean()
        )

        critic_loss = bellman_loss + self.cql_alpha * cql_loss

        self.critic_optim.zero_grad()
        if self.encoder_optim is not None:
            self.encoder_optim.zero_grad()
        critic_loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            if self.drift_encoder is not None:
                nn.utils.clip_grad_norm_(self.drift_encoder.parameters(), max_grad_norm)
        self.critic_optim.step()
        if self.encoder_optim is not None:
            self.encoder_optim.step()

        # --- Actor update (TD3+BC style when bc_weight > 0, pure CQL otherwise) ---
        context_detached = self._encode(drift).detach()
        pred_a = self.actor(states, context_detached)
        q_vals = self.critic.q_min(states, pred_a, context_detached)

        if self.bc_weight > 0:
            # TD3+BC: normalize Q scale so bc_weight is dataset-agnostic
            lam = self.bc_weight / (q_vals.abs().mean().detach() + 1e-8)
            bc_loss = F.mse_loss(pred_a, actions)
            actor_loss = -lam * q_vals.mean() + bc_loss
        else:
            actor_loss = -q_vals.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor_optim.step()

        # --- Soft update target critic ---
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "cql_loss": cql_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def select_action(
        self,
        state: np.ndarray,
        drift_features: np.ndarray,
    ) -> np.ndarray:
        """Select action given state and drift features."""
        self.actor.eval()
        if self.drift_encoder is not None:
            self.drift_encoder.eval()

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            d = torch.FloatTensor(drift_features).unsqueeze(0).to(self.device)
            context = self._encode(d)
            action = self.actor(s, context)

        return action.cpu().numpy().squeeze(0)

    def fit_context_distribution(self, drift_np: np.ndarray):
        """
        Fit Gaussian to training context distribution for Mahalanobis gate.
        Stores context_mu and context_sigma_inv. Call after encoder training.
        """
        if not self.use_drift:
            return
        self.drift_encoder.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(drift_np), 1024):
                d = torch.FloatTensor(drift_np[i:i + 1024]).to(self.device)
                chunks.append(self.drift_encoder(d).cpu().numpy())
        contexts = np.concatenate(chunks, axis=0)          # (N, context_dim)
        self.context_mu = contexts.mean(axis=0)            # (context_dim,)
        cov = np.cov(contexts.T) + 1e-6 * np.eye(contexts.shape[1])
        self.context_sigma_inv = np.linalg.inv(cov)        # (context_dim, context_dim)
        logger.info(
            f"Context distribution fit: mu={self.context_mu.round(3)}, "
            f"n={len(contexts)}"
        )

    def compute_context_distance(self, drift_np: np.ndarray) -> float:
        """Mahalanobis distance of current drift from training context distribution."""
        if not self.use_drift or not hasattr(self, "context_mu"):
            return 0.0
        self.drift_encoder.eval()
        with torch.no_grad():
            d = torch.FloatTensor(drift_np).unsqueeze(0).to(self.device)
            ctx = self.drift_encoder(d).cpu().numpy().squeeze(0)
        diff = ctx - self.context_mu
        dist_sq = float(diff @ self.context_sigma_inv @ diff)
        return float(np.sqrt(max(dist_sq, 0.0)))

    def save(self, path: str):
        """Save all SARC components."""
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "use_drift": self.use_drift,
        }
        if self.drift_encoder is not None:
            ckpt["drift_encoder"] = self.drift_encoder.state_dict()
        torch.save(ckpt, path)
        logger.info(f"SARC model saved to {path}")

    def load(self, path: str):
        """Load all SARC components."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        if self.drift_encoder is not None and "drift_encoder" in checkpoint:
            self.drift_encoder.load_state_dict(checkpoint["drift_encoder"])
        logger.info(f"SARC model loaded from {path}")
