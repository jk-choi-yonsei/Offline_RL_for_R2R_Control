"""Ground-truth Preston-simulator rollout evaluator.

Drives rollouts with a fresh ``CMPSimulator`` per test sequence. Each step
invokes the full Preston equation with zone-specific wear coupling, so the
wear-dependent asymmetry is part of the evaluation dynamics rather than
being averaged out by a wear-blind linear proxy.

Assumes:
  state  = (prev_RR, prev_action_1, ..., prev_action_K)
  action = (a_1, ..., a_K)
with the previous-action components of the state laid out contiguously at
``action_state_indices``.
"""

from typing import Dict, List

import numpy as np

from src.data.cmp_simulator import (
    CMPSimulator, SimConfig, ACTION_RAW_LOW, ACTION_RAW_HIGH,
)


class PrestonRolloutEvaluator:
    def __init__(
        self,
        cfg: SimConfig,
        scalers: dict,
        runs_per_lot: int,
        action_bounds,
        rr_state_idx: int = 0,
        action_state_indices=None,
        spec_margin_norm: float = 1.0,
    ):
        self.cfg = cfg
        self.state_scaler = scalers["state"]
        self.action_scaler = scalers["action"]
        self.drift_scaler = scalers["drift"]
        self.runs_per_lot = runs_per_lot
        self.lo, self.hi = action_bounds
        self.rr_idx = rr_state_idx
        if action_state_indices is None:
            # Default: contiguous block after rr_state_idx, length = action dim
            action_state_indices = tuple(
                range(rr_state_idx + 1, rr_state_idx + 1 + len(self.lo))
            )
        self.act_state_idx = tuple(action_state_indices)
        self.spec_margin = spec_margin_norm

        self._mu_rr = float(self.state_scaler.mean_[rr_state_idx])
        self._sig_rr = float(self.state_scaler.scale_[rr_state_idx])
        self._mu_act_state = np.array(
            [self.state_scaler.mean_[i] for i in self.act_state_idx],
            dtype=np.float64,
        )
        self._sig_act_state = np.array(
            [self.state_scaler.scale_[i] for i in self.act_state_idx],
            dtype=np.float64,
        )

    def _spawn_sim(self, seq_idx: int, d0_raw: np.ndarray) -> CMPSimulator:
        sim = CMPSimulator(self.cfg)
        sim.rng = np.random.default_rng(self.cfg.seed + seq_idx + 2026)
        # Resample base zone weights deterministically per sequence so all
        # controllers face the same underlying tool geometry.
        sim.base_zone_weights = sim.rng.dirichlet(
            np.ones(self.cfg.n_zones) * 5.0
        )
        sim.wear = np.clip(d0_raw[:4].astype(np.float64), 0.0, 1.0)
        sim._wear_rates = np.maximum(d0_raw[4:8].astype(np.float64), 1e-5)
        lot_pos0 = float(d0_raw[8])
        sim.lot_run = int(round(lot_pos0 * max(self.runs_per_lot - 1, 1)))
        return sim

    def evaluate(
        self,
        sequences: List[dict],
        controller,
        target_rr_norm: float = 0.0,
    ) -> Dict:
        all_errors: List[float] = []
        all_diffs: List[float] = []
        per_seq_mae: List[float] = []
        per_seq_ce: List[float] = []
        spec_viol = 0
        total = 0

        for seq_idx, seq in enumerate(sequences):
            controller.reset()

            d0_raw = self.drift_scaler.inverse_transform(
                seq["drift_features"][0:1]
            )[0]
            sim = self._spawn_sim(seq_idx, d0_raw)

            state = seq["observations"][0].copy()
            d = seq["drift_features"][0].copy()
            T = len(seq["observations"])

            seq_errors: List[float] = []
            seq_diffs: List[float] = []
            prev_action = None

            for t in range(T):
                action = controller.predict_action(state, d)
                action = np.clip(action, self.lo, self.hi)

                a_raw = self.action_scaler.inverse_transform(
                    action.reshape(1, -1)
                )[0]
                a_raw = np.clip(a_raw, ACTION_RAW_LOW, ACTION_RAW_HIGH).astype(np.float32)

                rr_raw, _ = sim.step(a_raw)
                rr_next_norm = (rr_raw - self._mu_rr) / self._sig_rr

                err = abs(rr_next_norm - target_rr_norm)
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

                controller.update(rr_next_norm, action)

                if t + 1 < T:
                    # State = [rr_norm, a_1_norm, ..., a_K_norm]. The action
                    # components of the state use the state scaler's mean/std
                    # for those positions (same normalization as training).
                    act_state_norm = (
                        a_raw.astype(np.float64) - self._mu_act_state
                    ) / self._sig_act_state
                    state = np.zeros_like(state)
                    state[self.rr_idx] = rr_next_norm
                    for i, idx in enumerate(self.act_state_idx):
                        state[idx] = float(act_state_norm[i])
                    dc_next = sim.get_drift_components(sim.lot_run, self.runs_per_lot)
                    next_d_raw = np.concatenate([
                        dc_next["wear"],
                        dc_next["delta_wear"],
                        [dc_next["lot_position"]],
                    ]).astype(np.float32)
                    d = self.drift_scaler.transform(
                        next_d_raw.reshape(1, -1)
                    )[0].astype(np.float32)

            per_seq_mae.append(float(np.mean(seq_errors)) if seq_errors else 0.0)
            per_seq_ce.append(float(np.mean(seq_diffs)) if seq_diffs else 0.0)

        return {
            "mae": float(np.mean(all_errors)),
            "rmse": float(np.sqrt(np.mean(np.array(all_errors) ** 2))),
            "action_cost": float(np.mean(all_diffs)) if all_diffs else 0.0,
            "spec_violation_rate": float(spec_viol / max(total, 1)),
            "n_transitions": total,
            "n_sequences": len(sequences),
            "per_sequence_mae": per_seq_mae,
            "per_sequence_ce": per_seq_ce,
        }
