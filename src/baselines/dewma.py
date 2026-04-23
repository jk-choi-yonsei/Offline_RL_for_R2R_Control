"""
Baseline R2R Controllers.
D-EWMA (Double Exponentially Weighted Moving Average) and variants.
"""

import numpy as np
from typing import List, Optional, Tuple


class EWMAController:
    """
    EWMA (Exponentially Weighted Moving Average) R2R Controller.

    Standard single-EWMA feedback controller. Estimates process bias via
    EWMA of observed RR, then adjusts action to compensate.

    Model: y_t = a_t + d_t  (gain=1 in normalized space)
    Update: d̂_t = λ * y_t + (1-λ) * d̂_{t-1}
    Control: a_{t+1} = -d̂_t  (cancel estimated bias)
    """

    def __init__(
        self,
        target_rr: float = 0.0,
        action_dim: int = 6,
        lam: float = 0.4,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.target_rr = target_rr
        self.action_dim = action_dim
        self.lam = lam

        self.d_hat = 0.0          # EWMA disturbance estimate
        self.last_action = np.zeros(action_dim)

        if action_bounds is not None:
            self.lower_bound, self.upper_bound = action_bounds
        else:
            self.lower_bound = np.full(action_dim, -3.0)
            self.upper_bound = np.full(action_dim,  3.0)

    def reset(self):
        """Reset to initial state (call at the start of each evaluation sequence)."""
        self.d_hat = 0.0
        self.last_action = np.zeros(self.action_dim)

    def update(self, actual_rr: float, action: np.ndarray):
        """Update disturbance estimate with observed RR."""
        error = actual_rr - self.target_rr
        self.d_hat = self.lam * error + (1 - self.lam) * self.d_hat
        self.last_action = action.copy()

    def predict_action(self, state: Optional[np.ndarray] = None, drift: Optional[np.ndarray] = None) -> np.ndarray:
        """Cancel estimated disturbance uniformly across all action dims."""
        correction = -self.d_hat
        action = self.last_action + correction * np.ones(self.action_dim)
        return np.clip(action, self.lower_bound, self.upper_bound)


class DEWMAController:
    """
    D-EWMA (Double Exponentially Weighted Moving Average) R2R Controller.

    Tracks process drift and adjusts control actions to maintain
    removal rate at target level.

    Model: y_t = β₀ + β₁ * x_t + ε_t
    EWMA update: β̂_t = λ * (y_t - β̂₁·x_t) + (1-λ) * β̂_{t-1}
    """

    def __init__(
        self,
        target_rr: float,
        action_dim: int = 5,
        lambda_0: float = 0.5,
        lambda_1: float = 0.3,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            target_rr: Target removal rate.
            action_dim: Number of action dimensions (zone pressures).
            lambda_0: EWMA weight for intercept.
            lambda_1: EWMA weight for slope.
            action_bounds: (lower, upper) bounds for actions.
        """
        self.target_rr = target_rr
        self.action_dim = action_dim
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1

        # Linear model parameters: y = beta0 + beta1 * x
        self.beta0 = np.zeros(1)  # Intercept estimate
        self.beta1 = np.ones(action_dim) / action_dim  # Slope estimate

        # Action bounds
        if action_bounds is not None:
            self.lower_bound, self.upper_bound = action_bounds
        else:
            self.lower_bound = np.ones(action_dim) * 0.1
            self.upper_bound = np.ones(action_dim) * 10.0

        self.history = {"actions": [], "predictions": [], "errors": []}

    def reset(self):
        """Reset to initial state (call at the start of each evaluation sequence)."""
        self.beta0 = np.zeros(1)
        self.beta1 = np.ones(self.action_dim) / self.action_dim
        self.history = {"actions": [], "predictions": [], "errors": []}

    def update(self, actual_rr: float, action: np.ndarray):
        """
        Update EWMA estimates based on observed outcome.

        Args:
            actual_rr: Observed removal rate.
            action: Action taken (zone pressures).
        """
        prediction = self.beta0 + np.dot(self.beta1, action)
        error = actual_rr - prediction

        # D-EWMA update
        self.beta0 = self.lambda_0 * (actual_rr - np.dot(self.beta1, action)) + \
                     (1 - self.lambda_0) * self.beta0
        self.beta1 = self.lambda_1 * (error * action / (np.dot(action, action) + 1e-8)) + \
                     (1 - self.lambda_1) * self.beta1

        self.history["actions"].append(action.copy())
        self.history["predictions"].append(prediction)
        self.history["errors"].append(error)

    def predict_action(self, state: Optional[np.ndarray] = None, drift: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute next action to achieve target RR.

        Uses current model estimates to solve for optimal action:
        target = beta0 + beta1^T * x  →  x = (target - beta0) / sum(beta1) * beta1/|beta1|

        Returns:
            Recommended action (zone pressures).
        """
        residual = self.target_rr - self.beta0
        beta_sum = np.sum(np.abs(self.beta1)) + 1e-8

        # Distribute action proportionally to beta1
        action = residual * self.beta1 / (np.dot(self.beta1, self.beta1) + 1e-8)

        # Clip to bounds
        action = np.clip(action, self.lower_bound, self.upper_bound)
        return action

    def run_episode(
        self,
        states: np.ndarray,
        true_rr: np.ndarray,
    ) -> dict:
        """
        Simulate controller over a sequence of runs.

        Args:
            states: State observations (N, state_dim).
            true_rr: True removal rates (N,).

        Returns:
            Dictionary with actions, predictions, errors.
        """
        actions = []
        predictions = []
        errors = []

        for t in range(len(states)):
            action = self.predict_action(states[t] if states is not None else None)
            actions.append(action)

            if t > 0:
                self.update(true_rr[t - 1], actions[-2])

            pred = float(self.beta0 + np.dot(self.beta1, action))
            predictions.append(pred)
            errors.append(self.target_rr - true_rr[t])

        return {
            "actions": np.array(actions),
            "predictions": np.array(predictions),
            "errors": np.array(errors),
            "true_rr": true_rr,
        }


class FTDEWMAController(DEWMAController):
    """
    FT-D-EWMA (Fine-Tuned D-EWMA) Controller.
    Adds priority-based cost adjustment.
    """

    def __init__(
        self,
        target_rr: float,
        action_dim: int = 5,
        lambda_0: float = 0.5,
        lambda_1: float = 0.3,
        cost_weights: Optional[np.ndarray] = None,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        super().__init__(target_rr, action_dim, lambda_0, lambda_1, action_bounds)
        self.cost_weights = cost_weights if cost_weights is not None else np.ones(action_dim)

    def predict_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cost-aware action.
        Prioritizes cheaper control variables.
        """
        residual = self.target_rr - self.beta0

        # Weight by inverse cost (cheaper actions used more)
        inv_cost = 1.0 / (self.cost_weights + 1e-8)
        weighted_beta = self.beta1 * inv_cost

        action = residual * weighted_beta / (np.dot(weighted_beta, self.beta1) + 1e-8)
        return np.clip(action, self.lower_bound, self.upper_bound)
