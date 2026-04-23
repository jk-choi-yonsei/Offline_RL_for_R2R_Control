"""
Kalman Filter based R2R Controller.
Uses Kalman Filter for process state estimation and control.
"""

import numpy as np
from typing import Optional, Tuple


class KalmanR2RController:
    """
    Kalman Filter R2R Controller.

    State-space model:
        x_{t+1} = A * x_t + B * u_t + w_t   (process model)
        y_t     = C * x_t + v_t              (observation model)

    Where:
        x: hidden process state (drift)
        u: control action (zone pressures)
        y: observed removal rate
    """

    def __init__(
        self,
        target_rr: float,
        state_dim: int = 2,   # [intercept, slope]
        action_dim: int = 5,
        process_noise: float = 0.01,
        measurement_noise: float = 1.0,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.target_rr = target_rr
        self.kf_state_dim = state_dim
        self.action_dim = action_dim

        # Kalman state: [beta0, beta1_mean]
        self.x = np.zeros(state_dim)  # State estimate
        self.x[0] = target_rr  # Initial intercept = target

        # State transition (random walk model for drift)
        self.A = np.eye(state_dim)

        # Covariances
        self.P = np.eye(state_dim) * 100.0     # State covariance (high initial uncertainty)
        self.Q = np.eye(state_dim) * process_noise  # Process noise
        self.R = np.eye(1) * measurement_noise       # Measurement noise

        # Effective beta1 for action mapping
        self.beta1 = np.ones(action_dim) / action_dim

        if action_bounds is not None:
            self.lower_bound, self.upper_bound = action_bounds
        else:
            self.lower_bound = np.ones(action_dim) * 0.1
            self.upper_bound = np.ones(action_dim) * 10.0

        self.history = {"states": [], "P_trace": [], "innovations": []}

    def reset(self):
        """Reset to initial state (call at the start of each evaluation sequence)."""
        self.x = np.zeros(self.kf_state_dim)
        self.x[0] = self.target_rr
        self.P = np.eye(self.kf_state_dim) * 100.0
        self.beta1 = np.ones(self.action_dim) / self.action_dim
        self.history = {"states": [], "P_trace": [], "innovations": []}

    def _predict(self):
        """Kalman time-update (predict) step — called internally by predict_action."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def predict(self):
        """Public alias for backward compatibility."""
        self._predict()

    def update(self, actual_rr: float, action: np.ndarray):
        """
        Kalman update step.

        Args:
            actual_rr: Observed removal rate.
            action: Action taken.
        """
        # Observation model: y = C * x where C = [1, sum(action)]
        C = np.array([[1.0, np.mean(action)]])
        y = np.array([actual_rr])

        # Innovation
        y_pred = C @ self.x
        innovation = y - y_pred

        # Kalman gain
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + (K @ innovation).flatten()
        self.P = (np.eye(self.kf_state_dim) - K @ C) @ self.P

        self.history["states"].append(self.x.copy())
        self.history["P_trace"].append(np.trace(self.P))
        self.history["innovations"].append(float(innovation))

    def predict_action(self, state: Optional[np.ndarray] = None, drift: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run Kalman predict step then compute action based on state estimate.
        The predict step is folded in so callers don't need to call predict() manually.
        """
        self._predict()
        # Current drift estimate
        intercept = self.x[0]
        slope = self.x[1] if self.kf_state_dim > 1 else 1.0

        residual = self.target_rr - intercept
        action = residual * self.beta1 / (slope * np.sum(self.beta1 ** 2) + 1e-8)

        return np.clip(action, self.lower_bound, self.upper_bound)

    def run_episode(
        self,
        states: np.ndarray,
        true_rr: np.ndarray,
    ) -> dict:
        """
        Simulate Kalman controller over a sequence.

        Args:
            states: State observations (N, state_dim).
            true_rr: True removal rates (N,).

        Returns:
            Results dictionary.
        """
        actions = []
        predictions = []
        errors = []

        for t in range(len(true_rr)):
            # predict_action() calls _predict() internally
            action = self.predict_action(states[t] if states is not None else None)
            actions.append(action)

            # Predicted RR
            pred = float(self.x[0] + self.x[1] * np.mean(action)) if self.kf_state_dim > 1 \
                   else float(self.x[0])
            predictions.append(pred)

            # Update with observation
            self.update(true_rr[t], action)

            errors.append(self.target_rr - true_rr[t])

        return {
            "actions": np.array(actions),
            "predictions": np.array(predictions),
            "errors": np.array(errors),
            "true_rr": true_rr,
            "kf_states": np.array(self.history["states"]),
            "P_trace": np.array(self.history["P_trace"]),
        }
