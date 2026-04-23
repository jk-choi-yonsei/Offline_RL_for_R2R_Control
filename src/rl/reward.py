"""
Unified reward function for R2R Control.

r = -alpha * (error / spec_margin)^2 - lambda_s * ||Da||_1
Clipped to [-10, 0].

Used by all datasets (CMP2, CMP1, Simulator) with dataset-specific
target_rr and spec_margin values.
"""

import numpy as np
from typing import Optional


def compute_reward(
    actual_rr: float,
    target_rr: float,
    action_t: np.ndarray,
    action_prev: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    lambda_s: float = 0.01,
    spec_margin: float = 1.0,
) -> float:
    """
    Compute unified R2R control reward (scalar).

    Args:
        actual_rr: Achieved removal rate.
        target_rr: Target removal rate.
        action_t: Current action vector.
        action_prev: Previous action vector (None for first step).
        alpha: Tracking accuracy weight.
        lambda_s: Action smoothness weight.
        spec_margin: Specification half-width (sigma_spec).

    Returns:
        Scalar reward in [-10, 0].
    """
    error = actual_rr - target_rr
    r = -alpha * (error / spec_margin) ** 2

    if action_prev is not None:
        r -= lambda_s * float(np.sum(np.abs(action_t - action_prev)))

    return float(np.clip(r, -10.0, 0.0))


def compute_reward_batch(
    actual_rr: np.ndarray,
    target_rr: float,
    actions: np.ndarray,
    alpha: float = 1.0,
    lambda_s: float = 0.01,
    spec_margin: float = 1.0,
) -> np.ndarray:
    """
    Compute unified reward for a batch of transitions.

    Args:
        actual_rr: Array of achieved RR. Shape: (N,).
        target_rr: Target removal rate.
        actions: Action array. Shape: (N, action_dim).
        alpha: Tracking accuracy weight.
        lambda_s: Action smoothness weight.
        spec_margin: Specification half-width.

    Returns:
        Rewards. Shape: (N,).
    """
    actual_rr = np.atleast_1d(actual_rr).astype(np.float64)
    error = actual_rr - target_rr

    # Tracking term
    r = -alpha * (error / spec_margin) ** 2

    # Action smoothness term
    actions = np.atleast_2d(actions)
    action_prev = np.vstack([actions[0:1], actions[:-1]])
    delta_action = np.abs(actions - action_prev)
    r -= lambda_s * np.sum(delta_action, axis=-1)

    return np.clip(r, -10.0, 0.0).astype(np.float32)
