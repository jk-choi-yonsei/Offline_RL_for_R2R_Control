"""
Noise disturbance models for R2R control evaluation.

IMA(0,1,1) and ARIMA(1,1,1) processes simulate non-stationary
equipment drift / degradation, as used in semiconductor R2R control literature.

Reference setup: noise resets every `reset_interval` steps to simulate
                 scheduled equipment maintenance.
"""

import numpy as np


class IMANoise:
    """
    IMA(0,1,1) disturbance:
        d_t = d_{t-1} + e_t - theta * e_{t-1},  e_t ~ N(0, sigma^2)

    The cumulative disturbance d_t is added to the process output (RR).
    Models slow, non-stationary equipment drift.
    """

    def __init__(self, sigma: float = 0.3, theta: float = 0.5, reset_interval: int = 100):
        self.sigma = sigma
        self.theta = theta
        self.reset_interval = reset_interval
        self.reset()

    def reset(self):
        self.d = 0.0
        self.prev_e = 0.0
        self.step = 0

    def sample(self) -> float:
        if self.reset_interval > 0 and self.step > 0 and self.step % self.reset_interval == 0:
            self.reset()
        e = np.random.normal(0.0, self.sigma)
        self.d = self.d + e - self.theta * self.prev_e
        self.prev_e = e
        self.step += 1
        return self.d


class ARIMANoise:
    """
    ARIMA(1,1,1) disturbance:
        Delta(y_t) = phi * Delta(y_{t-1}) + e_t - theta * e_{t-1}
        y_t        = y_{t-1} + Delta(y_t),  e_t ~ N(0, sigma^2)

    Models auto-correlated, non-stationary process drift.
    """

    def __init__(
        self,
        sigma: float = 0.3,
        phi: float = 0.5,
        theta: float = 0.5,
        reset_interval: int = 100,
    ):
        self.sigma = sigma
        self.phi = phi
        self.theta = theta
        self.reset_interval = reset_interval
        self.reset()

    def reset(self):
        self.y = 0.0
        self.prev_dy = 0.0
        self.prev_e = 0.0
        self.step = 0

    def sample(self) -> float:
        if self.reset_interval > 0 and self.step > 0 and self.step % self.reset_interval == 0:
            self.reset()
        e = np.random.normal(0.0, self.sigma)
        dy = self.phi * self.prev_dy + e - self.theta * self.prev_e
        self.y += dy
        self.prev_dy = dy
        self.prev_e = e
        self.step += 1
        return self.y


def build_noise_model(noise_type: str, sigma: float, reset_interval: int = 100):
    """Factory function."""
    if noise_type == "none" or noise_type is None:
        return None
    elif noise_type == "ima":
        return IMANoise(sigma=sigma, theta=0.5, reset_interval=reset_interval)
    elif noise_type == "arima":
        return ARIMANoise(sigma=sigma, phi=0.5, theta=0.5, reset_interval=reset_interval)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Use 'none', 'ima', or 'arima'.")


def prefetch_noise(noise_model, n_steps: int, seed: int = 42) -> np.ndarray:
    """
    Pre-generate a fixed noise sequence so that all controllers in one
    evaluation run see exactly the same disturbance realization.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    noise_model.reset()
    seq = np.array([noise_model.sample() for _ in range(n_steps)])
    np.random.set_state(rng_state)
    noise_model.reset()
    return seq
