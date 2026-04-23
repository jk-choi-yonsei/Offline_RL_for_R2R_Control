"""
Preston CMP simulator -> offline RL MDP dataset (6D action: P_1..P_5, V).

Preston with zone-specific wear coupling + velocity:
  RR = k_p(w_pad, w_mem) * sum_i w_i(w_dresser) * P_i * V + noise

EWMA's uniform scalar correction cannot track zone-asymmetric drift driven
by dresser wear; the DriftEncoder, conditioned on the wear vector, can. See
cmp_simulator.py for the physical motivation.

MDP dimensions:
  State  7D: prev_RR + prev_P_1..P_5 + prev_V   (wear unobserved)
  Action 6D: (P_1, ..., P_5, V), StandardScaler-normalized
  Drift  9D: wear(4) + delta_wear(4) + lot_position(1)
  Reward: -alpha*(e/sigma)^2 - lambda_s * ||da||_1
"""

import os
import sys
import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.cmp_simulator import (
    CMPSimulator, SimConfig, ACTION_RAW_LOW, ACTION_RAW_HIGH,
)
from src.data.mdp_dataset import split_data, save_dataset
from src.data.config import RESULTS_DIR

logger = logging.getLogger(__name__)

N_ZONES = 5
ACTION_DIM = N_ZONES + 1          # (P_1..P_5, V)
STATE_DIM = 1 + N_ZONES + 1       # (prev_RR, prev_P_1..P_5, prev_V)

# Nominal action: uniform mid-pressure + nominal velocity.
ACTION_NOMINAL = np.concatenate([
    np.full(N_ZONES, (ACTION_RAW_LOW[0] + ACTION_RAW_HIGH[0]) / 2.0),
    [(ACTION_RAW_LOW[N_ZONES] + ACTION_RAW_HIGH[N_ZONES]) / 2.0],
]).astype(np.float32)


def compute_reward(
    rr: float,
    prev_action: np.ndarray,
    action: np.ndarray,
    target: float,
    spec_margin: float,
    lambda_s: float = 0.01,
) -> float:
    from src.rl.reward import compute_reward as _unified_reward
    return _unified_reward(rr, target, action, prev_action,
                           lambda_s=lambda_s, spec_margin=spec_margin)


class TunedEWMABehavioral:
    """Drift-unaware EWMA behavioral policy.

    Computes a single scalar EWMA correction from RR error; applies it
    uniformly to all zone pressures (mirrors industrial global-EWMA
    practice). Velocity V is held at its nominal with exploration noise.

    Coverage on zone-specific variation comes from per-zone Gaussian noise;
    that is the only source of signal the RL learner has for untangling the
    asymmetric wear response.
    """

    def __init__(
        self,
        target: float,
        k0_nominal: float,
        n_zones: int = N_ZONES,
        lam: float = 0.6,
        noise_p: float = 0.25,
        noise_v: float = 0.08,
        p_base: Optional[float] = None,
        v_base: float = 1.0,
        rng=None,
    ):
        self.target = target
        self.k0_nominal = k0_nominal
        self.n_zones = n_zones
        self.lam = lam
        self.noise_p = noise_p
        self.noise_v = noise_v
        self.v_base = v_base
        self.p_base = (target / (k0_nominal * v_base)) if p_base is None else p_base
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.ewma_err = 0.0

    def predict_action(self) -> np.ndarray:
        correction_p = self.ewma_err / (self.k0_nominal * self.v_base)
        p_mean = self.p_base + correction_p
        p_vec = np.full(self.n_zones, p_mean) + self.rng.normal(
            0.0, self.noise_p, self.n_zones
        )
        v = self.v_base + self.rng.normal(0.0, self.noise_v)
        a = np.concatenate([p_vec, [v]]).astype(np.float32)
        return np.clip(a, ACTION_RAW_LOW, ACTION_RAW_HIGH).astype(np.float32)

    def update(self, rr: float):
        err = self.target - rr
        self.ewma_err = self.lam * err + (1.0 - self.lam) * self.ewma_err


class NominalController:
    """Legacy proportional + EWMA controller (kept as alt behavioral)."""

    def __init__(self, target: float, k0: float, n_zones: int = N_ZONES,
                 alpha: float = 0.4, noise_p: float = 0.15, noise_v: float = 0.05,
                 v_base: float = 1.0, rng=None):
        self.target = target
        self.k0_est = k0
        self.n_zones = n_zones
        self.alpha = alpha
        self.noise_p = noise_p
        self.noise_v = noise_v
        self.v_base = v_base
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.rr_ewma = target

    def predict_action(self) -> np.ndarray:
        p_nominal = self.target / (self.k0_est * self.v_base)
        correction = (self.target - self.rr_ewma) / (self.k0_est * self.v_base) * 0.6
        p_mean = p_nominal + correction
        p_vec = np.full(self.n_zones, p_mean) + self.rng.normal(
            0.0, self.noise_p, self.n_zones
        )
        v = self.v_base + self.rng.normal(0.0, self.noise_v)
        a = np.concatenate([p_vec, [v]]).astype(np.float32)
        return np.clip(a, ACTION_RAW_LOW, ACTION_RAW_HIGH).astype(np.float32)

    def update(self, rr: float):
        self.rr_ewma = self.alpha * rr + (1 - self.alpha) * self.rr_ewma


def generate_dataset(
    n_lots: int = 400,
    runs_per_lot: int = 20,
    drift_scale: float = 1.0,
    seed: int = 42,
    tag: str = "sim",
    lambda_s: float = 0.01,
    behavioral: str = "tuned_ewma",
) -> dict:
    """Generate raw MDP data: state=(prev_RR, prev_P_1..5, prev_V), action=(P_1..5, V)."""
    cfg = SimConfig(drift_scale=drift_scale, seed=seed)
    sim = CMPSimulator(cfg)

    target = cfg.target_rr
    spec_margin = cfg.spec_margin

    obs_list, act_list, rew_list = [], [], []
    next_obs_list, term_list = [], []
    drift_list, next_drift_list = [], []

    prev_wear = None

    for lot_idx in range(n_lots):
        sim.reset_lot(inherit_wear=prev_wear)

        if behavioral == "tuned_ewma":
            ctrl = TunedEWMABehavioral(
                target=target, k0_nominal=cfg.k0, n_zones=N_ZONES,
                lam=0.6, noise_p=0.25, noise_v=0.08,
                rng=np.random.default_rng(seed + lot_idx),
            )
        elif behavioral == "nominal":
            ctrl = NominalController(
                target=target, k0=cfg.k0, n_zones=N_ZONES,
                alpha=0.4, noise_p=0.15, noise_v=0.05,
                rng=np.random.default_rng(seed + lot_idx),
            )
        else:
            raise ValueError(f"Unknown behavioral='{behavioral}'")

        prev_rr = target
        prev_action = ACTION_NOMINAL.copy()

        lot_obs, lot_act, lot_rew = [], [], []
        lot_next_obs, lot_drift, lot_next_drift = [], [], []

        for run_idx in range(runs_per_lot - 1):
            dc = sim.get_drift_components(run_idx, runs_per_lot)
            state_raw = np.concatenate([
                [prev_rr], prev_action.astype(np.float32),
            ]).astype(np.float32)
            drift_raw = np.concatenate([
                dc["wear"], dc["delta_wear"], [dc["lot_position"]],
            ]).astype(np.float32)

            action_raw = ctrl.predict_action().astype(np.float32)
            rr_next, _ = sim.step(action_raw)

            dc_next = sim.get_drift_components(run_idx + 1, runs_per_lot)
            next_state_raw = np.concatenate([
                [rr_next], action_raw.astype(np.float32),
            ]).astype(np.float32)
            next_drift_raw = np.concatenate([
                dc_next["wear"], dc_next["delta_wear"], [dc_next["lot_position"]],
            ]).astype(np.float32)

            reward = compute_reward(
                rr_next, prev_action, action_raw, target, spec_margin,
                lambda_s=lambda_s,
            )

            ctrl.update(rr_next)

            lot_obs.append(state_raw)
            lot_act.append(action_raw)
            lot_rew.append(np.float32(reward))
            lot_next_obs.append(next_state_raw)
            lot_drift.append(drift_raw)
            lot_next_drift.append(next_drift_raw)

            prev_rr = rr_next
            prev_action = action_raw.copy()

        n_lot = len(lot_obs)
        if n_lot > 0:
            terminals = np.zeros(n_lot, dtype=np.float32)
            terminals[-1] = 1.0

            obs_list.extend(lot_obs)
            act_list.extend(lot_act)
            rew_list.extend(lot_rew)
            next_obs_list.extend(lot_next_obs)
            drift_list.extend(lot_drift)
            next_drift_list.extend(lot_next_drift)
            term_list.extend(terminals.tolist())

        prev_wear = sim.wear.copy()

    mdp_raw = {
        "observations":        np.array(obs_list,        dtype=np.float32),
        "actions":             np.array(act_list,        dtype=np.float32),
        "rewards":             np.array(rew_list,        dtype=np.float32),
        "next_observations":   np.array(next_obs_list,   dtype=np.float32),
        "terminals":           np.array(term_list,       dtype=np.float32),
        "drift_features":      np.array(drift_list,      dtype=np.float32),
        "next_drift_features": np.array(next_drift_list, dtype=np.float32),
    }

    n = len(obs_list)
    rr_col = mdp_raw["observations"][:, 0]
    p_col = mdp_raw["actions"][:, :N_ZONES]
    v_col = mdp_raw["actions"][:, N_ZONES]
    logger.info(
        f"[{tag}] {n} transitions ({n_lots} lots x ~{runs_per_lot-1} runs) | "
        f"RR mean={rr_col.mean():.1f} std={rr_col.std():.1f} | "
        f"P mean={p_col.mean():.2f} std={p_col.std():.2f} | "
        f"V mean={v_col.mean():.2f} std={v_col.std():.2f}"
    )
    return mdp_raw


def normalize_and_split(mdp_raw: dict, tag: str = "sim"):
    s_scaler = StandardScaler()
    a_scaler = StandardScaler()
    d_scaler = StandardScaler()

    mdp = dict(mdp_raw)
    mdp["observations"]        = s_scaler.fit_transform(mdp["observations"]).astype(np.float32)
    mdp["next_observations"]   = s_scaler.transform(mdp["next_observations"]).astype(np.float32)
    mdp["actions"]             = a_scaler.fit_transform(mdp["actions"]).astype(np.float32)
    mdp["drift_features"]      = d_scaler.fit_transform(mdp["drift_features"]).astype(np.float32)
    mdp["next_drift_features"] = d_scaler.transform(mdp["next_drift_features"]).astype(np.float32)

    train_data, val_data, test_data = split_data(mdp)
    scalers = {"state": s_scaler, "action": a_scaler, "drift": d_scaler}

    logger.info(
        f"[{tag}] split: train={len(train_data['observations'])}, "
        f"val={len(val_data['observations'])}, "
        f"test={len(test_data['observations'])}"
    )
    return train_data, val_data, test_data, scalers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    for drift_scale, tag in [(0.4, "sim_mild"), (1.0, "sim_medium"), (2.0, "sim_heavy")]:
        logger.info(f"Generating {tag} (drift_scale={drift_scale})...")
        mdp_raw = generate_dataset(
            n_lots=400, runs_per_lot=20,
            drift_scale=drift_scale, seed=42, tag=tag,
        )
        train, val, test, scalers = normalize_and_split(mdp_raw, tag=tag)
        save_dataset(train, scalers,      {"target_rr": 0.0}, f"{tag}_train")
        save_dataset(val,   {},           {"target_rr": 0.0}, f"{tag}_val")
        save_dataset(test,  {},           {"target_rr": 0.0}, f"{tag}_test")

    logger.info("Done. Files saved to results/data/sim_*")
