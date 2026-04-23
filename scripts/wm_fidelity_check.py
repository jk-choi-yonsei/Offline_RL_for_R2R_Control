"""Quick WM fidelity diagnostic for CMP1 and CMP2.

Motivated by the sim linear-proxy bug: the sim evaluator used a wear-blind linear
model, so SARC's wear-aware advantage was erased at eval time. CMP1/CMP2 use a
neural WM (5-ensemble, delta prediction), and `world_model.py` itself flags
"weak action signal (corr ~0.10) from the dominant state-persistence term." If
the WM has collapsed toward state persistence, an RL policy's counterfactual
action won't move predicted RR much, and all methods will bunch up near the
behavioral baseline — exactly the "SARC ~ BC ~ NoDrift" pattern we see.

This script quantifies, for each dataset:
  1. WM next-RR MAE on the held-out test split (sanity).
  2. Action sensitivity: fix state, sweep action along each zone by ±1σ, measure
     Δ(predicted RR). Compare to the empirical std of RR on the test split.
  3. Counterfactual range: for each test transition, sample K random actions
     from the action distribution (policy-agnostic) and measure the spread of
     predicted RR. If spread << data std, WM is ignoring action.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.config import RESULTS_DIR
from src.models.world_model import WorldModel


DATA_DIR = os.path.join(RESULTS_DIR, "data")


def load_split(tag: str, split: str):
    keys = ["observations", "actions", "next_observations", "rewards"]
    return {k: np.load(os.path.join(DATA_DIR, f"{tag}_{split}_{k}.npy")) for k in keys}


def wm_rr_pred(wm: WorldModel, states: np.ndarray, actions: np.ndarray, rr_idx):
    """Return predicted next-RR (mean over rr_idx slice)."""
    next_s, _, _ = wm.predict(states.astype(np.float32), actions.astype(np.float32), deterministic=True)
    if isinstance(rr_idx, tuple):
        s, e = rr_idx
        return next_s[:, s:e].mean(axis=1)
    return next_s[:, rr_idx]


def diagnose(tag: str, rr_idx, wm_ckpt_name: str):
    train = load_split(tag, "train")
    test = load_split(tag, "test")

    state_dim = train["observations"].shape[1]
    action_dim = train["actions"].shape[1]
    wm = WorldModel(state_dim=state_dim, action_dim=action_dim, device="cpu")
    wm._load_checkpoint(wm_ckpt_name)
    wm.ensemble.eval()

    # Test set RR stats (normalized)
    if isinstance(rr_idx, tuple):
        s, e = rr_idx
        rr_next_true = test["next_observations"][:, s:e].mean(axis=1)
        rr_prev = test["observations"][:, s:e].mean(axis=1)
    else:
        rr_next_true = test["next_observations"][:, rr_idx]
        rr_prev = test["observations"][:, rr_idx]

    rr_std = rr_next_true.std()
    delta_rr_std = (rr_next_true - rr_prev).std()

    # 1) WM MAE on next-RR
    rr_pred = wm_rr_pred(wm, test["observations"], test["actions"], rr_idx)
    wm_mae = np.mean(np.abs(rr_pred - rr_next_true))

    # Baseline: predict "next_RR = prev_RR" (state persistence)
    persist_mae = np.mean(np.abs(rr_prev - rr_next_true))

    # 2) Action sensitivity: for a subsample, vary each action dim by ±1σ
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(test["observations"]), size=min(500, len(test["observations"])), replace=False)
    s_sub = test["observations"][idx_sub]
    a_sub = test["actions"][idx_sub]

    action_std = test["actions"].std(axis=0)
    # sweep ±1σ along each dim separately
    sens_per_dim = []
    for d in range(action_dim):
        a_plus = a_sub.copy();  a_plus[:, d]  += action_std[d]
        a_minus = a_sub.copy(); a_minus[:, d] -= action_std[d]
        rr_plus = wm_rr_pred(wm, s_sub, a_plus, rr_idx)
        rr_minus = wm_rr_pred(wm, s_sub, a_minus, rr_idx)
        sens_per_dim.append(np.mean(np.abs(rr_plus - rr_minus)))

    mean_sens = float(np.mean(sens_per_dim))

    # 3) Counterfactual spread: sample 16 random actions per state, measure std
    K = 16
    cf_stds = []
    for i in idx_sub:
        s_i = np.tile(test["observations"][i], (K, 1))
        # sample from empirical action distribution
        sample_idx = rng.integers(0, len(train["actions"]), size=K)
        a_i = train["actions"][sample_idx]
        rr_k = wm_rr_pred(wm, s_i, a_i, rr_idx)
        cf_stds.append(rr_k.std())
    cf_spread = float(np.mean(cf_stds))

    # state-only baseline: zero action (centered post-StandardScaler)
    a_zero = np.zeros_like(a_sub)
    rr_zero = wm_rr_pred(wm, s_sub, a_zero, rr_idx)
    rr_real = wm_rr_pred(wm, s_sub, a_sub, rr_idx)
    zero_vs_real = float(np.mean(np.abs(rr_zero - rr_real)))

    print(f"\n{'=' * 60}")
    print(f"WM fidelity diagnostic -- {tag}")
    print(f"{'=' * 60}")
    print(f"Test size: {len(test['observations'])}  State={state_dim}D  Action={action_dim}D")
    print(f"RR std (next, norm):         {rr_std:.4f}")
    print(f"dRR std (next-prev, norm):   {delta_rr_std:.4f}")
    print(f"WM next-RR MAE:              {wm_mae:.4f}")
    print(f"Persistence baseline MAE:    {persist_mae:.4f}  "
          f"(WM gain: {(persist_mae - wm_mae) / persist_mae * 100:+.1f}%)")
    print(f"Action +/-1sigma sensitivity:{mean_sens:.4f}  "
          f"(per zone; vs dRR std {delta_rr_std:.4f})")
    print(f"Counterfactual RR spread:    {cf_spread:.4f}  (K=16 random train actions)")
    print(f"Zero-action vs real-action:  {zero_vs_real:.4f}  "
          f"(if ~0, WM ignores action magnitude)")
    return {
        "tag": tag, "wm_mae": wm_mae, "persist_mae": persist_mae,
        "action_sens_per_dim": sens_per_dim, "cf_spread": cf_spread,
        "zero_vs_real": zero_vs_real, "rr_std": float(rr_std),
        "delta_rr_std": float(delta_rr_std),
    }


if __name__ == "__main__":
    # CMP2: state last 4 dims are RR channels (AKE3, AK, AKE1, AKE2)
    r2 = diagnose("cmp2", rr_idx=(9, 13), wm_ckpt_name="cmp2_best")
    # CMP1: last 1 dim is prev_RR
    r1 = diagnose("cmp1", rr_idx=-1, wm_ckpt_name="cmp1_best")

    print("\n" + "=" * 60)
    print("Summary: is WM ignoring actions?")
    print("=" * 60)
    for r in [r2, r1]:
        ratio = r["cf_spread"] / max(r["rr_std"], 1e-6)
        verdict = "OK" if ratio > 0.25 else ("SUSPECT" if ratio > 0.1 else "BROKEN")
        print(f"{r['tag']}: CF spread / RR std = {ratio:.3f}  -> {verdict}")
