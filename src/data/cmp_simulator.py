"""
Preston-equation CMP simulator with zone-specific wear coupling.

  RR = k_p(w_pad, w_mem) * sum_i w_i(w_dresser) * P_i * V + noise
  k_p(w_pad, w_mem) = k0 * exp(-alpha_pad * w_pad) * (1 + gamma * w_mem^2)
  w_i(w_dresser)    = softmax_i( log(base_w_i) + beta_i * w_dresser )

Physical motivation (CMP literature):
  - Pad wear reduces Preston coefficient uniformly (standard, e.g. Luo 2001).
  - Dresser wear causes *zone-specific* conditioning asymmetry; center
    zones drift up in effectiveness, edge zones drift down (or vice versa),
    driven by pad surface roughness nonuniformity. This is the center-to-
    edge nonuniformity routinely reported in CMP reviews (Zhao 2013,
    Suzuki 2017). beta_i = [+2, +1, 0, -1, -2] encodes a monotonic
    center-to-edge asymmetry pattern.

Why this breaks EWMA:
  EWMA applies a single scalar correction across all action dims:
    a_{t+1} = a_t + d_hat * ones
  When w_dresser drifts, different zones need different corrections. A
  uniform shift cannot simultaneously raise P_1 and lower P_5. The
  DriftEncoder, which sees w_dresser directly, can learn this asymmetric
  recipe and dominate the baseline.

Action is 6D: (P_1, P_2, P_3, P_4, P_5, V).
  P_i: zone i down force (psi)
  V:   platen velocity (normalized around 1.0)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SimConfig:
    n_zones: int = 5
    target_rr: float = 2850.0
    spec_margin: float = 450.0
    k0: float = 1140.0              # chosen so k0 * P_c * V_c ≈ target at nominal
    alpha_pad: float = 0.8          # pad wear → k_p decay
    gamma_membrane: float = 0.2     # membrane wear → mild k_p nonlinearity
    # Zone asymmetry coefficient vector — center-to-edge monotonic pattern.
    # As w_dresser rises, inner zones gain weight, outer zones lose (or vice
    # versa depending on sign convention). Magnitudes control how much the
    # zone mix shifts across a lot.
    zone_beta: tuple = (2.0, 1.0, 0.0, -1.0, -2.0)
    noise_std: float = 80.0

    # Action bounds (raw, unnormalized)
    p_min: float = 0.5
    p_max: float = 5.0
    v_min: float = 0.5
    v_max: float = 2.0

    # Wear rates
    wear_rate_pad: float = 0.015
    wear_rate_dresser: float = 0.010
    wear_rate_membrane: float = 0.007
    wear_rate_retainer: float = 0.005
    wear_reset_prob: float = 1.00
    drift_scale: float = 1.0
    seed: int = 42


ACTION_RAW_LOW = np.array(
    [SimConfig.p_min] * SimConfig.n_zones + [SimConfig.v_min],
    dtype=np.float32,
)
ACTION_RAW_HIGH = np.array(
    [SimConfig.p_max] * SimConfig.n_zones + [SimConfig.v_max],
    dtype=np.float32,
)


class CMPSimulator:
    """Preston simulator with zone-specific wear coupling and (P_zones, V) action."""

    def __init__(self, config: SimConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.wear = np.zeros(4)
        self._wear_rates = np.zeros(4)
        self.base_zone_weights = np.ones(config.n_zones) / config.n_zones
        self.lot_run = 0
        self.reset_lot()

    def reset_lot(self, inherit_wear: Optional[np.ndarray] = None):
        cfg = self.cfg

        base_rates = np.array([
            cfg.wear_rate_pad,
            cfg.wear_rate_dresser,
            cfg.wear_rate_membrane,
            cfg.wear_rate_retainer,
        ])
        self._wear_rates = (
            base_rates
            * cfg.drift_scale
            * (1.0 + 0.2 * self.rng.standard_normal(4))
        ).clip(min=1e-5)

        if inherit_wear is not None and self.rng.random() > cfg.wear_reset_prob:
            self.wear = inherit_wear.copy()
        else:
            self.wear = np.clip(0.05 * self.rng.random(4), 0.0, 1.0)

        # Dirichlet base weights give per-lot variation around the uniform mix.
        # The dresser-driven drift is applied on top of this every step.
        self.base_zone_weights = self.rng.dirichlet(
            np.ones(cfg.n_zones) * 5.0
        )
        self.lot_run = 0

    def k_p(self) -> float:
        cfg = self.cfg
        w = self.wear
        kp = (
            cfg.k0
            * np.exp(-cfg.alpha_pad * w[0])
            * (1.0 + cfg.gamma_membrane * w[2] ** 2)
        )
        return float(np.clip(kp, cfg.k0 * 0.05, cfg.k0 * 1.5))

    def zone_weights(self) -> np.ndarray:
        """Dresser-wear-dependent per-zone weights (sum to 1)."""
        cfg = self.cfg
        w_dresser = float(self.wear[1])
        logits = np.log(self.base_zone_weights + 1e-8) + np.array(cfg.zone_beta) * w_dresser
        logits -= logits.max()  # numerical stability
        w = np.exp(logits)
        return w / w.sum()

    def step(self, action_raw: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        One run simulation.

        Args:
            action_raw: shape (n_zones + 1,). First n_zones entries are zone
                pressures (psi); the last entry is velocity V (normalized).

        Returns:
            rr: removal rate (Å)
            wear: wear vector copy after update
        """
        action_raw = np.asarray(action_raw, dtype=np.float64).reshape(-1)
        n_zones = self.cfg.n_zones
        p_vec = action_raw[:n_zones]
        v = float(action_raw[n_zones])

        kp = self.k_p()
        zw = self.zone_weights()
        rr = kp * float(np.dot(zw, p_vec)) * v
        rr += self.rng.normal(0.0, self.cfg.noise_std)
        rr = float(np.clip(rr, 0.0, 6000.0))

        self.wear = np.clip(self.wear + self._wear_rates, 0.0, 1.0)
        self.lot_run += 1

        return rr, self.wear.copy()

    def get_drift_components(self, lot_run_idx: int, runs_per_lot: int) -> dict:
        lot_pos = lot_run_idx / max(runs_per_lot - 1, 1)
        return {
            "wear": self.wear.copy(),
            "delta_wear": self._wear_rates.copy(),
            "lot_position": float(lot_pos),
        }


if __name__ == "__main__":
    sim = CMPSimulator(SimConfig(drift_scale=1.0))
    sim.reset_lot()
    print("run | RR(A)  | k_p   | w_pad | w_dres | zone_weights")
    print("-" * 75)
    for i in range(15):
        action = np.concatenate([np.full(5, 2.5), [1.0]])
        rr, wear = sim.step(action)
        zw = sim.zone_weights()
        print(f"{i:>3} | {rr:>6.1f} | {sim.k_p():>5.1f} | {wear[0]:.3f} | {wear[1]:.3f}  | "
              f"[{zw[0]:.2f} {zw[1]:.2f} {zw[2]:.2f} {zw[3]:.2f} {zw[4]:.2f}]")
