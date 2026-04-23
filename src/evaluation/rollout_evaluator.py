"""
Unified rollout evaluation framework for R2R controllers.

Defines a common BaseController interface and RolloutEvaluator so that
train_sarc_cmp1.py and train_sarc_sim.py share one evaluation loop instead
of copy-pasted evaluate_controller / ewma_based_eval / kalman_based_eval.

Usage:
    evaluator = RolloutEvaluator(action_bounds=NORM_BOUNDS, rr_state_idx=-1)
    controllers = {
        "SARC":          SARCController(agent),
        "SARC-no-drift": NoDriftController(agent, context_dim),
        "BC":            BCController(bc_agent),
        "D-EWMA":        DEWMAController(target_rr, action_dim, action_bounds=NORM_BOUNDS),
        "Kalman":        KalmanR2RController(target_rr, action_dim, action_bounds=NORM_BOUNDS),
    }
    for name, ctrl in controllers.items():
        metrics[name] = evaluator.evaluate(sequences, dynamics_fn, ctrl, target_rr,
                                           noise_reset_fn=_reset_noise)
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ============================================================
# Base interface
# ============================================================

class BaseController(ABC):
    """Protocol for all R2R controllers used in rollout evaluation."""

    def reset(self):
        """Reset stateful components. Called at the start of each sequence."""
        pass

    @abstractmethod
    def predict_action(self, state: np.ndarray, drift: np.ndarray) -> np.ndarray:
        """Return normalized action given current state and drift features."""
        ...

    def update(self, rr: float, action: np.ndarray):
        """Update internal state after observing RR outcome (stateful controllers)."""
        pass


# ============================================================
# RL agent wrappers
# ============================================================

class SARCController(BaseController):
    """Wraps SARCAgent for rollout evaluation.

    Works for both use_drift=True and use_drift=False agents.
    Delegates to agent.select_action() which handles drift_encoder=None internally.
    """

    def __init__(self, agent):
        self.agent = agent

    def predict_action(self, state: np.ndarray, drift: np.ndarray) -> np.ndarray:
        self.agent.actor.eval()
        if self.agent.drift_encoder is not None:
            self.agent.drift_encoder.eval()
        return self.agent.select_action(state, drift)


class NoDriftController(BaseController):
    """SARC ablation: zeroed drift context (tests DriftEncoder contribution)."""

    def __init__(self, agent, context_dim: int):
        self.agent = agent
        self.context_dim = context_dim

    def predict_action(self, state: np.ndarray, drift: np.ndarray) -> np.ndarray:
        self.agent.actor.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            ctx = torch.zeros(1, self.context_dim).to(self.agent.device)
            a = self.agent.actor(s, ctx)
        return a.cpu().numpy().squeeze(0)


class BCController(BaseController):
    """Wraps BCAgent for rollout evaluation."""

    def __init__(self, agent):
        self.agent = agent

    def predict_action(self, state: np.ndarray, drift: np.ndarray) -> np.ndarray:
        return self.agent.select_action(state, drift)


class GatedSARCController(BaseController):
    """
    SARC with Mahalanobis-distance gate.

    Blends SARC action with a D-EWMA fallback based on how novel the
    current drift context is relative to the training distribution.

        alpha = sigmoid(-sharpness * (distance - threshold))

    High distance (OOD drift) → low alpha → more D-EWMA.
    Low distance  (familiar)  → high alpha → pure SARC.

    threshold and sharpness are calibrated from validation-set distances:
        threshold = percentile(val_distances, pct)
        sharpness = 1 / std(val_distances)
    """

    def __init__(self, agent, dewma_controller, threshold: float, sharpness: float):
        self.agent = agent
        self.dewma = dewma_controller
        self.threshold = threshold
        self.sharpness = sharpness

    def reset(self):
        self.dewma.reset()

    def predict_action(self, state: np.ndarray, drift: np.ndarray) -> np.ndarray:
        self.agent.actor.eval()
        if self.agent.drift_encoder is not None:
            self.agent.drift_encoder.eval()

        a_rl   = self.agent.select_action(state, drift)
        a_ewma = self.dewma.predict_action(state)

        dist  = self.agent.compute_context_distance(drift)
        z     = -self.sharpness * (dist - self.threshold)
        alpha = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

        return alpha * a_rl + (1.0 - alpha) * a_ewma

    def update(self, rr: float, action: np.ndarray):
        self.dewma.update(rr, action)


# ============================================================
# Evaluator
# ============================================================

class RolloutEvaluator:
    """
    Evaluates any BaseController on a list of test sequences via
    open-loop dynamics simulation.

    Handles both stateless (SARC, BC) and stateful (EWMA, Kalman) controllers
    uniformly: controller.reset() is called at the start of every sequence,
    controller.update() after every step.
    """

    def __init__(
        self,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        rr_state_idx: int = -1,
        rr_state_end: Optional[int] = None,
        spec_margin_norm: float = 1.0,
    ):
        """
        Args:
            action_bounds: (lower, upper) arrays for clipping actions.
            rr_state_idx:  Start index of the RR dimension(s) in the state vector.
            rr_state_end:  End index (exclusive) for multi-RR state update.
                           When set, state[rr_state_idx:rr_state_end] = rr_next.
                           When None, state[rr_state_idx] = rr_next (single dim).
            spec_margin_norm: Spec limit in normalized RR units.
        """
        self.lo, self.hi = action_bounds
        self.rr_idx = rr_state_idx
        self.rr_end = rr_state_end
        self.spec_margin = spec_margin_norm

    def evaluate(
        self,
        sequences: List[dict],
        dynamics_fn: Callable,
        controller: BaseController,
        target_rr: float = 0.0,
        noise_reset_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Simulate controller rollouts on test sequences.

        Args:
            sequences:      List of sequence dicts (observations, actions,
                            drift_features, next_observations, next_drift_features).
            dynamics_fn:    Callable(state, action) -> rr_next (float).
            controller:     Any BaseController instance.
            target_rr:      Target RR in normalized units (0.0 = dataset mean).
            noise_reset_fn: Optional callable to reset shared noise counter so
                            every controller sees the same disturbance sequence.

        Returns:
            Dict with mae, rmse, action_cost, spec_violation_rate,
            n_transitions, n_sequences.
        """
        if noise_reset_fn is not None:
            noise_reset_fn()

        all_errors: List[float] = []
        all_diffs:  List[float] = []
        per_seq_mae: List[float] = []
        per_seq_ce:  List[float] = []
        spec_viol = 0
        total = 0

        for seq in sequences:
            controller.reset()

            obs   = seq["observations"]
            drift = seq.get("drift_features", np.zeros((len(obs), 1)))
            T     = len(obs)

            seq_errors: List[float] = []
            seq_diffs:  List[float] = []

            prev_action = None
            state = obs[0].copy()
            d     = drift[0].copy()

            for t in range(T):
                action = controller.predict_action(state, d)
                action = np.clip(action, self.lo, self.hi)

                rr_next = dynamics_fn(state, action)
                err     = abs(rr_next - target_rr)
                all_errors.append(err)
                seq_errors.append(err)
                if err > self.spec_margin:
                    spec_viol += 1
                total += 1

                if prev_action is not None:
                    diff = float(np.mean(np.abs(action - prev_action)))
                    all_diffs.append(diff)
                    seq_diffs.append(diff)
                prev_action = action.copy()

                controller.update(rr_next, action)

                if t + 1 < T:
                    next_s = seq["next_observations"][t].copy()
                    if self.rr_end is not None:
                        next_s[self.rr_idx:self.rr_end] = rr_next
                    else:
                        next_s[self.rr_idx] = rr_next
                    state = next_s
                    if "next_drift_features" in seq:
                        d = seq["next_drift_features"][t].copy()

            per_seq_mae.append(float(np.mean(seq_errors)) if seq_errors else 0.0)
            per_seq_ce.append(float(np.mean(seq_diffs)) if seq_diffs else 0.0)

        return {
            "mae":                float(np.mean(all_errors)),
            "rmse":               float(np.sqrt(np.mean(np.array(all_errors) ** 2))),
            "action_cost":        float(np.mean(all_diffs)) if all_diffs else 0.0,
            "spec_violation_rate": float(spec_viol / max(total, 1)),
            "n_transitions":      total,
            "n_sequences":        len(sequences),
            "per_sequence_mae":   per_seq_mae,
            "per_sequence_ce":    per_seq_ce,
        }
