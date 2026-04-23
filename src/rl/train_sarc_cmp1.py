"""
SARC Training and Evaluation on CMP1 Dataset.

CMP1 is a public PHM dataset used to demonstrate generalizability.
State: 13D (wear_last x6 + process_mean x6 + prev_RR x1)
Action: 6D (zone pressures)
Drift: 13D (wear x6 + delta_wear x6 + lot_position x1)

Evaluation uses a Neural World Model (5-ensemble) trained on CMP1 training data
to simulate RR outcomes for each controller on test sequences.

Paper final CMP1: cql_alpha=5.0, bc_weight=0.5, context_dim=2,
pretrain=80, CQL=200, lr=1e-4, batch=256, seeds [123,789,1024,2024,7777].
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.config import RESULTS_DIR, CHECKPOINT_DIR, WorldModelConfig
from src.models.world_model import WorldModel
from src.data.preprocess_cmp1_r2r import preprocess_cmp1_r2r
from src.data.mdp_dataset import split_data, chrono_split_data, save_dataset
from src.rl.sarc_agent import SARCAgent
from src.rl.bc_agent import BCAgent
from src.baselines.dewma import DEWMAController
from src.baselines.kalman import KalmanR2RController
from src.evaluation.noise_models import build_noise_model, prefetch_noise
from src.evaluation.rollout_evaluator import (
    RolloutEvaluator, SARCController, BCController,
)

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
STATE_DIM = 13   # wear(6) + process(6) + prev_RR(1)
ACTION_DIM = 6   # 6 zone pressures
DRIFT_DIM = 13   # wear(6) + delta_wear(6) + lot_position(1)
RR_OUT_DIM = 1   # single RR channel in CMP1
RR_STATE_IDX = -1  # prev_RR is the last dim of state

# Normalized action bounds for baselines (3-sigma)
NORM_BOUNDS = (np.full(ACTION_DIM, -3.0), np.full(ACTION_DIM, 3.0))


# ============================================================
# Baseline hyperparameter tuning on validation set
# Industrial selection criterion: val_MAE + AC_WEIGHT * val_action_cost.
# ============================================================
AC_WEIGHT = 0.0


def tune_dewma(val_sequences, dynamics_fn, target_rr_norm, evaluator) -> dict:
    """Grid search lambda_0 x lambda_1 for D-EWMA (joint criterion)."""
    best_score, best_params = float("inf"), (0.5, 0.3)
    logger.info("Tuning D-EWMA lambdas on validation set ...")
    for l0 in [0.2, 0.3, 0.5]:
        for l1 in [0.5, 0.6, 0.7]:
            ctrl = DEWMAController(
                target_rr=target_rr_norm, action_dim=ACTION_DIM,
                lambda_0=l0, lambda_1=l1, action_bounds=NORM_BOUNDS,
            )
            res = evaluator.evaluate(val_sequences, dynamics_fn, ctrl, target_rr_norm)
            score = res["mae"] + AC_WEIGHT * res["action_cost"]
            if score < best_score:
                best_score = score
                best_params = (l0, l1)
    logger.info(f"Best D-EWMA: lambda_0={best_params[0]}, lambda_1={best_params[1]} "
                f"(val score={best_score:.4f})")
    return {"lambda_0": best_params[0], "lambda_1": best_params[1]}


# ============================================================
# Step 2: Extract test sequences
# ============================================================

def extract_test_sequences(test_data: dict):
    """
    Group test transitions into sequences using terminal flags.
    Each sequence dict includes 'stage' (0=A, 1=B) from the first transition.
    """
    terminals = test_data["terminals"]
    has_stage = "stage_labels" in test_data
    n = len(terminals)

    sequences = []
    seq_start = 0

    for i in range(n):
        if terminals[i] == 1.0 or i == n - 1:
            seq_end = i + 1
            if seq_end - seq_start >= 2:
                seq = {k: v[seq_start:seq_end] for k, v in test_data.items()}
                if has_stage:
                    seq["stage"] = int(test_data["stage_labels"][seq_start])
                sequences.append(seq)
            seq_start = seq_end

    n_a = sum(1 for s in sequences if s.get("stage", -1) == 0)
    n_b = sum(1 for s in sequences if s.get("stage", -1) == 1)
    logger.info(f"Extracted {len(sequences)} test sequences "
                f"(avg length: {np.mean([len(s['observations']) for s in sequences]):.1f}, "
                f"Stage A: {n_a}, Stage B: {n_b})")
    return sequences


# ============================================================
# Step 3: Evaluate a controller on test sequences
# ============================================================

def evaluate_controller(
    sequences: list,
    dynamics_fn,
    get_action_fn,
    target_rr_normalized: float = 0.0,
) -> dict:
    """
    Simulate controller rollouts and compute metrics.

    Args:
        sequences: List of test sequence dicts.
        dynamics_fn: Callable(state, action) -> rr_next (float).
        get_action_fn: Callable(state, drift) -> normalized action (6D).
        target_rr_normalized: Target RR in normalized units (0.0 = dataset mean).
    """
    all_rr_errors = []
    all_action_diffs = []
    spec_violations = 0
    total_steps = 0

    SPEC_MARGIN_NORM = 1.0

    for seq in sequences:
        obs = seq["observations"].copy()
        drift = seq["drift_features"].copy()
        T = len(obs)

        prev_action = None
        state = obs[0].copy()
        d = drift[0].copy()

        for t in range(T):
            action = get_action_fn(state, d)
            action = np.clip(action, NORM_BOUNDS[0], NORM_BOUNDS[1])

            rr_next = dynamics_fn(state, action)

            rr_error = abs(rr_next - target_rr_normalized)
            all_rr_errors.append(rr_error)

            if rr_error > SPEC_MARGIN_NORM:
                spec_violations += 1
            total_steps += 1

            if prev_action is not None:
                all_action_diffs.append(np.mean(np.abs(action - prev_action)))
            prev_action = action.copy()

            if t + 1 < T:
                next_state = seq["next_observations"][t].copy()
                next_state[RR_STATE_IDX] = rr_next  # inject simulated RR
                state = next_state
                d = seq["next_drift_features"][t].copy()

    return {
        "mae": float(np.mean(all_rr_errors)),
        "rmse": float(np.sqrt(np.mean(np.array(all_rr_errors) ** 2))),
        "action_cost": float(np.mean(all_action_diffs)) if all_action_diffs else 0.0,
        "spec_violation_rate": float(spec_violations / max(total_steps, 1)),
        "n_transitions": total_steps,
        "n_sequences": len(sequences),
    }


# ============================================================
# Main Training & Evaluation Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SARC on CMP1 dataset")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cql-alpha", type=float, default=5.0,
                        help="Paper final CMP1: alpha_CQL=5.0 (via run_unified_phase1.py).")
    parser.add_argument("--context-dim", type=int, default=2)
    parser.add_argument("--bc-weight", type=float, default=0.5,
                        help="Behavioral cloning weight for actor loss (TD3+BC). 0=pure CQL.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing CMP1 model for evaluation")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (save final model only)")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Skip DriftEncoder pretraining (ablation: random encoder init)")
    parser.add_argument("--raw-drift", action="store_true",
                        help="Bypass DriftEncoder: use raw 13D drift features directly as context")
    parser.add_argument("--model-suffix", type=str, default="",
                        help="Suffix for checkpoint and results filenames (e.g. '_nopretrain')")
    parser.add_argument("--only-nodrift", action="store_true",
                        help="Re-train only the SARC-no-drift (use_drift=False) ablation. "
                             "Skips main SARC/BC; patches the existing JSON's SARC-no-drift entries.")
    parser.add_argument("--use-wm", action="store_true",
                        help="[deprecated -- Neural World Model is always used]")
    parser.add_argument("--noise-type", type=str, default="none",
                        choices=["none", "ima", "arima"],
                        help="Noise disturbance model added to simulated RR (default: none)")
    parser.add_argument("--noise-scale", type=float, default=0.3,
                        help="Noise std sigma in normalized RR units (default: 0.3)")
    parser.add_argument("--noise-reset-interval", type=int, default=100,
                        help="Steps between noise resets (simulates equipment maintenance)")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Random seed for pre-generating noise sequence")
    parser.add_argument("--train-noise-scale", type=float, default=0.0,
                        help="Gaussian noise sigma added to RR dim during CQL training. 0=disabled.")
    parser.add_argument("--chrono-split", action="store_true",
                        help="Chronological lot split (early lots->train, late lots->test). "
                             "Tests distribution shift generalization under increasing equipment wear.")
    parser.add_argument("--cross-lot", action="store_true",
                        help="Cross-lot sequences: group by (STAGE, CHAMBER) across all 185 lots.")
    parser.add_argument("--lambda-s", type=float, default=0.01,
                        help="Action smoothness weight in reward (default: 0.01)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ============================================================
    # 1. Preprocess CMP1
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 1: Preprocess CMP1 R2R Data")
    logger.info("=" * 60)

    mdp_data, target_rr_dict, s_scaler, a_scaler = preprocess_cmp1_r2r(
        cross_lot=args.cross_lot, lambda_s=args.lambda_s,
    )

    if args.chrono_split:
        train_data, val_data, test_data = chrono_split_data(mdp_data)
        logger.info("Using CHRONOLOGICAL split (early lots->train, late lots->test)")
    else:
        train_data, val_data, test_data = split_data(mdp_data)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"Splits - Train: {len(train_data['observations'])}, "
                f"Val: {len(val_data['observations'])}, "
                f"Test: {len(test_data['observations'])}")
    logger.info(f"State: {STATE_DIM}D, Action: {ACTION_DIM}D, Drift: {DRIFT_DIM}D")

    save_dataset(train_data, {"state": s_scaler, "action": a_scaler}, target_rr_dict, "cmp1_train")
    save_dataset(val_data, {}, target_rr_dict, "cmp1_val")
    save_dataset(test_data, {}, target_rr_dict, "cmp1_test")

    # ============================================================
    # 2. Load Neural World Model
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 2: Load Neural World Model (for offline eval)")
    logger.info("=" * 60)

    target_rr_norm = 0.0

    wm_ckpt = os.path.join(CHECKPOINT_DIR, "world_model_cmp1_best.pt")
    if not os.path.exists(wm_ckpt):
        raise FileNotFoundError(
            f"CMP1 World Model not found at {wm_ckpt}. "
            "Run: python src/models/train_world_model_cmp1.py first."
        )
    eval_wm = WorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, device="cpu")
    eval_wm._load_checkpoint("cmp1_best")
    eval_wm.ensemble.eval()
    logger.info(f"Using neural World Model for evaluation: {wm_ckpt}")

    def _base_dynamics(state, action):
        next_s, _, _ = eval_wm.predict(
            state.reshape(1, -1).astype(np.float32),
            action.reshape(1, -1).astype(np.float32),
            deterministic=True,
        )
        return float(next_s[0, RR_STATE_IDX])

    noise_model = build_noise_model(
        args.noise_type, args.noise_scale, args.noise_reset_interval
    )
    if noise_model is not None:
        _total_test_steps = sum(len(s["observations"]) for s in extract_test_sequences(test_data))
        _noise_seq = prefetch_noise(noise_model, _total_test_steps + 1000, seed=args.noise_seed)
        _noise_idx = [0]

        def dynamics_fn(state, action):
            rr = _base_dynamics(state, action)
            n = _noise_seq[_noise_idx[0] % len(_noise_seq)]
            _noise_idx[0] += 1
            return rr + n

        logger.info(
            f"Noise injection: type={args.noise_type}, sigma={args.noise_scale}, "
            f"reset_interval={args.noise_reset_interval}"
        )
    else:
        def dynamics_fn(state, action):
            return _base_dynamics(state, action)

        _noise_idx = None

    def _reset_noise():
        if _noise_idx is not None:
            _noise_idx[0] = 0

    # ============================================================
    # Short-circuit: --only-nodrift path
    # ============================================================
    if args.only_nodrift:
        logger.info("=" * 60)
        logger.info("[--only-nodrift] Re-train SARC-no-drift ablation (use_drift=False)")
        logger.info("=" * 60)

        save_dir = os.path.join(RESULTS_DIR, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        suffix = args.model_suffix
        if args.seed != 42 or "_s" in suffix:
            suffix = suffix + f"_s{args.seed}"
        no_drift_ckpt = os.path.join(save_dir, f"sarc_cmp1_model_nodrift{suffix}.pt")
        no_drift_best = no_drift_ckpt.replace(".pt", "_best.pt")

        no_drift_agent = SARCAgent(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, drift_dim=DRIFT_DIM,
            context_dim=args.context_dim, lr_actor=args.lr, lr_critic=args.lr,
            cql_alpha=args.cql_alpha, rr_out_dim=RR_OUT_DIM,
            bc_weight=args.bc_weight, use_drift=False, device=args.device,
        )
        dev = no_drift_agent.device

        states_t = torch.FloatTensor(train_data["observations"]).to(dev)
        actions_t = torch.FloatTensor(train_data["actions"]).to(dev)
        rewards_t = torch.FloatTensor(train_data["rewards"]).unsqueeze(-1).to(dev)
        next_states_t = torch.FloatTensor(train_data["next_observations"]).to(dev)
        terminals_t = torch.FloatTensor(train_data["terminals"]).unsqueeze(-1).to(dev)
        drift_t = torch.FloatTensor(train_data["drift_features"]).to(dev)
        next_drift_t = torch.FloatTensor(train_data["next_drift_features"]).to(dev)
        n_samples = len(states_t)

        val_sequences = extract_test_sequences(val_data)
        val_evaluator = RolloutEvaluator(NORM_BOUNDS, rr_state_idx=RR_STATE_IDX)

        best_nd_mae = float("inf")
        for epoch in range(args.epochs):
            no_drift_agent.actor.train()
            perm = np.random.permutation(n_samples)
            for start in range(0, n_samples, args.batch_size):
                b = perm[start:start + args.batch_size]
                if len(b) < 8:
                    continue
                no_drift_agent.update(
                    states_t[b], actions_t[b], rewards_t[b],
                    next_states_t[b], terminals_t[b],
                    drift_t[b], next_drift_t[b],
                )
            if (epoch + 1) % 20 == 0:
                no_drift_agent.actor.eval()
                if len(val_sequences) > 0:
                    nd_val = val_evaluator.evaluate(
                        val_sequences, dynamics_fn,
                        SARCController(no_drift_agent), target_rr_norm,
                        noise_reset_fn=_reset_noise,
                    )
                    if nd_val["mae"] < best_nd_mae:
                        best_nd_mae = nd_val["mae"]
                        no_drift_agent.save(no_drift_best)
                    if (epoch + 1) % 40 == 0:
                        logger.info(f"  [no-drift] Epoch {epoch+1}/{args.epochs} | ValMAE={nd_val['mae']:.4f}")

        if os.path.exists(no_drift_best):
            no_drift_agent.load(no_drift_best)
            logger.info(f"Loaded best no-drift model (val MAE={best_nd_mae:.4f})")
        no_drift_agent.save(no_drift_ckpt)
        logger.info(f"SARC-no-drift model saved to {no_drift_ckpt}")

        test_sequences = extract_test_sequences(test_data)
        evaluator = RolloutEvaluator(NORM_BOUNDS, rr_state_idx=RR_STATE_IDX)
        nd_ctrl = SARCController(no_drift_agent)

        logger.info("Evaluating SARC-no-drift on test sequences ...")
        nd_test = evaluator.evaluate(
            test_sequences, dynamics_fn, nd_ctrl, target_rr_norm,
            noise_reset_fn=_reset_noise,
        )
        stage_seqs = {
            "A": [s for s in test_sequences if s.get("stage", -1) == 0],
            "B": [s for s in test_sequences if s.get("stage", -1) == 1],
        }
        nd_per_stage = {}
        if stage_seqs["A"] and stage_seqs["B"]:
            for stage_name, seqs in stage_seqs.items():
                nd_per_stage[stage_name] = evaluator.evaluate(
                    seqs, dynamics_fn, nd_ctrl, target_rr_norm, _reset_noise,
                )

        results_path = os.path.join(RESULTS_DIR, f"cmp1_evaluation{suffix}.json")
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"Expected existing JSON {results_path} to patch. "
                "Run the full pipeline once before --only-nodrift."
            )
        with open(results_path, "r") as f:
            existing = json.load(f)
        existing["methods"]["SARC-no-drift"] = nd_test
        if nd_per_stage and "per_stage" in existing:
            for stage_name, stage_res in nd_per_stage.items():
                existing["per_stage"][stage_name]["SARC-no-drift"] = stage_res
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2)

        print("\n" + "=" * 70)
        print(f"SARC-no-drift re-trained (use_drift=False) - seed={args.seed}")
        print("=" * 70)
        print(f"Overall: MAE={nd_test['mae']:.4f} RMSE={nd_test['rmse']:.4f} "
              f"AC={nd_test['action_cost']:.4f}")
        for stage_name, stage_res in nd_per_stage.items():
            print(f"Stage {stage_name}: MAE={stage_res['mae']:.4f} "
                  f"AC={stage_res['action_cost']:.4f}")
        logger.info(f"JSON patched: {results_path}")
        return existing

    # ============================================================
    # 3. Create SARC Agent
    # ============================================================
    from sklearn.preprocessing import StandardScaler
    drift_scaler = None
    if args.raw_drift:
        drift_scaler = StandardScaler()
        train_data["drift_features"] = drift_scaler.fit_transform(train_data["drift_features"])
        val_data["drift_features"] = drift_scaler.transform(val_data["drift_features"])
        test_data["drift_features"] = drift_scaler.transform(test_data["drift_features"])
        train_data["next_drift_features"] = drift_scaler.transform(train_data["next_drift_features"])
        val_data["next_drift_features"] = drift_scaler.transform(val_data["next_drift_features"])
        test_data["next_drift_features"] = drift_scaler.transform(test_data["next_drift_features"])
        logger.info("Raw drift mode: drift features normalized via StandardScaler (fair comparison)")

    effective_context_dim = DRIFT_DIM if args.raw_drift else args.context_dim

    agent = SARCAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        drift_dim=DRIFT_DIM,
        context_dim=effective_context_dim,
        lr_actor=args.lr,
        lr_critic=args.lr,
        cql_alpha=args.cql_alpha,
        rr_out_dim=RR_OUT_DIM,
        bc_weight=args.bc_weight,
        device=args.device,
    )

    if args.raw_drift:
        agent.drift_encoder = torch.nn.Identity()
        logger.info("Raw drift mode: DriftEncoder replaced with Identity (13D pass-through)")

    save_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    suffix = args.model_suffix
    if args.seed != 42 or "_s" in suffix:
        suffix = suffix + f"_s{args.seed}"
    cmp1_model_path = os.path.join(save_dir, f"sarc_cmp1_model{suffix}.pt")
    nodrift_model_path = os.path.join(save_dir, f"sarc_cmp1_model_nodrift{suffix}.pt")
    nodrift_best_model_path = nodrift_model_path.replace(".pt", "_best.pt")
    bc_model_path = os.path.join(save_dir, f"bc_cmp1_model{suffix}.pt")

    bc_agent = BCAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        drift_dim=DRIFT_DIM,
        context_dim=effective_context_dim,
        hidden_dim=256,
        lr=args.lr,
        lr_encoder=args.lr,
        rr_out_dim=RR_OUT_DIM,
        use_drift=(not args.raw_drift),
        device=args.device,
    )
    if args.raw_drift:
        bc_agent.drift_encoder = None

    no_drift_agent = SARCAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, drift_dim=DRIFT_DIM,
        context_dim=args.context_dim, lr_actor=args.lr, lr_critic=args.lr,
        cql_alpha=args.cql_alpha, rr_out_dim=RR_OUT_DIM,
        bc_weight=args.bc_weight, use_drift=False, device=args.device,
    )

    _sarc_best = cmp1_model_path.replace(".pt", "_best.pt")
    _bc_best   = bc_model_path.replace(".pt", "_best.pt")
    if args.skip_train and (os.path.exists(cmp1_model_path) or os.path.exists(_sarc_best)):
        _load = _sarc_best if os.path.exists(_sarc_best) else cmp1_model_path
        logger.info(f"Loading existing CMP1 model from {_load}")
        agent.load(_load)
        _bc_load = _bc_best if os.path.exists(_bc_best) else bc_model_path
        if os.path.exists(_bc_load):
            logger.info(f"Loading existing BC model from {_bc_load}")
            bc_agent.load(_bc_load)
        _nd_load = nodrift_best_model_path if os.path.exists(nodrift_best_model_path) else nodrift_model_path
        if os.path.exists(_nd_load):
            logger.info(f"Loading existing no-drift model from {_nd_load}")
            no_drift_agent.load(_nd_load)
    else:
        # ============================================================
        # 4. Pre-train Drift Encoder (1D RR prediction for CMP1)
        # ============================================================
        if args.raw_drift:
            logger.info("=" * 60)
            logger.info("Step 3: Skipping DriftEncoder pretrain (--raw-drift: identity encoder)")
            logger.info("=" * 60)
        elif args.no_pretrain:
            logger.info("=" * 60)
            logger.info("Step 3: Skipping DriftEncoder pretrain (--no-pretrain flag)")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("Step 3: Pre-train Drift Encoder (1D RR auxiliary task)")
            logger.info("=" * 60)

            train_drift = torch.FloatTensor(train_data["drift_features"])
            train_rr = torch.FloatTensor(train_data["observations"][:, -1:])  # (N, 1)

            pretrain_losses = agent.pretrain_encoder(
                train_drift, train_rr,
                n_epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
            )
            logger.info(f"Encoder pretrain complete. Final loss: {pretrain_losses[-1]:.6f}")

        # ============================================================
        # 5. CQL Training
        # ============================================================
        logger.info("=" * 60)
        logger.info("Step 4: SARC CQL Training on CMP1")
        logger.info("=" * 60)

        states = torch.FloatTensor(train_data["observations"]).to(agent.device)
        actions = torch.FloatTensor(train_data["actions"]).to(agent.device)
        rewards = torch.FloatTensor(train_data["rewards"]).unsqueeze(-1).to(agent.device)
        next_states = torch.FloatTensor(train_data["next_observations"]).to(agent.device)
        terminals = torch.FloatTensor(train_data["terminals"]).unsqueeze(-1).to(agent.device)
        drift = torch.FloatTensor(train_data["drift_features"]).to(agent.device)
        next_drift = torch.FloatTensor(train_data["next_drift_features"]).to(agent.device)

        n_samples = len(states)
        best_val_mae = float("inf")
        best_model_path = cmp1_model_path.replace(".pt", f"_best.pt")

        val_sequences = extract_test_sequences(val_data)

        def _sarc_action_fn(s, d):
            with torch.no_grad():
                st = torch.FloatTensor(s).unsqueeze(0).to(agent.device)
                dt = torch.FloatTensor(d).unsqueeze(0).to(agent.device)
                ctx = agent.drift_encoder(dt)
                a = agent.actor(st, ctx)
            return a.cpu().numpy().squeeze(0)

        for epoch in range(args.epochs):
            agent.actor.train()
            agent.drift_encoder.train()
            indices = np.random.permutation(n_samples)
            epoch_losses = {"critic_loss": [], "actor_loss": [], "cql_loss": []}

            for start in range(0, n_samples, args.batch_size):
                idx = indices[start:start + args.batch_size]
                if len(idx) < 8:
                    continue

                if args.train_noise_scale > 0.0:
                    s_aug = states[idx].clone()
                    ns_aug = next_states[idx].clone()
                    rr_noise = torch.randn(len(idx), device=agent.device) * args.train_noise_scale
                    s_aug[:, RR_STATE_IDX]  += rr_noise
                    ns_aug[:, RR_STATE_IDX] += rr_noise
                    batch_states      = s_aug
                    batch_next_states = ns_aug
                else:
                    batch_states      = states[idx]
                    batch_next_states = next_states[idx]

                losses = agent.update(
                    batch_states, actions[idx], rewards[idx],
                    batch_next_states, terminals[idx],
                    drift[idx], next_drift[idx],
                )
                for k in epoch_losses:
                    if k in losses:
                        epoch_losses[k].append(losses[k])

            if (epoch + 1) % 20 == 0:
                agent.actor.eval()
                agent.drift_encoder.eval()

                if not args.no_early_stop and len(val_sequences) > 0:
                    val_metrics = evaluate_controller(
                        val_sequences, dynamics_fn, _sarc_action_fn, target_rr_norm
                    )
                    val_mae = val_metrics["mae"]
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        agent.save(best_model_path)
                        logger.info(f"  -> New best val MAE: {val_mae:.4f} (saved)")

                crit = np.mean(epoch_losses["critic_loss"]) if epoch_losses["critic_loss"] else 0
                act = np.mean(epoch_losses["actor_loss"]) if epoch_losses["actor_loss"] else 0
                cql = np.mean(epoch_losses["cql_loss"]) if epoch_losses["cql_loss"] else 0
                logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                            f"Critic: {crit:.4f} | Actor: {act:.4f} | CQL: {cql:.4f}")

        if not args.no_early_stop and os.path.exists(best_model_path):
            logger.info(f"Loading best CMP1 model (val MAE={best_val_mae:.4f})")
            agent.load(best_model_path)
        agent.save(cmp1_model_path)
        logger.info(f"CMP1 SARC model saved to {cmp1_model_path}")

        # ============================================================
        # Step 4a: SARC-no-drift Ablation (use_drift=False, separately trained)
        # ============================================================
        logger.info("=" * 60)
        logger.info("Step 4a: SARC-no-drift Training (use_drift=False)")
        logger.info("=" * 60)

        best_nd_mae = float("inf")
        for epoch in range(args.epochs):
            no_drift_agent.actor.train()
            perm = np.random.permutation(n_samples)
            for start in range(0, n_samples, args.batch_size):
                b = perm[start:start + args.batch_size]
                if len(b) < 8:
                    continue
                no_drift_agent.update(
                    states[b], actions[b], rewards[b],
                    next_states[b], terminals[b],
                    drift[b], next_drift[b],
                )
            if (epoch + 1) % 20 == 0:
                no_drift_agent.actor.eval()
                if not args.no_early_stop and len(val_sequences) > 0:
                    def _nd_action_fn(s, d):
                        return no_drift_agent.select_action(s, d)
                    nd_val = evaluate_controller(
                        val_sequences, dynamics_fn, _nd_action_fn, target_rr_norm
                    )
                    if nd_val["mae"] < best_nd_mae:
                        best_nd_mae = nd_val["mae"]
                        no_drift_agent.save(nodrift_best_model_path)
                        logger.info(f"  -> [no-drift] New best val MAE: {nd_val['mae']:.4f} (saved)")

        if not args.no_early_stop and os.path.exists(nodrift_best_model_path):
            logger.info(f"Loading best no-drift model (val MAE={best_nd_mae:.4f})")
            no_drift_agent.load(nodrift_best_model_path)
        no_drift_agent.save(nodrift_model_path)
        logger.info(f"CMP1 SARC-no-drift model saved to {nodrift_model_path}")

        # ============================================================
        # BC Training
        # ============================================================
        logger.info("=" * 60)
        logger.info("Step 4b: BC Training on CMP1 (MSE imitation)")
        logger.info("=" * 60)

        if not args.raw_drift and not args.no_pretrain:
            bc_agent.pretrain_encoder(
                torch.FloatTensor(train_data["drift_features"]),
                torch.FloatTensor(train_data["observations"][:, -1:]),
                n_epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
            )

        bc_best_val_mae = float("inf")
        bc_best_model_path = bc_model_path.replace(".pt", "_best.pt")

        for epoch in range(args.epochs):
            indices = np.random.permutation(n_samples)
            epoch_bc_loss = []
            for start in range(0, n_samples, args.batch_size):
                idx = indices[start:start + args.batch_size]
                if len(idx) < 8:
                    continue
                loss = bc_agent.train_step(states[idx], actions[idx], drift[idx])
                epoch_bc_loss.append(loss)

            if (epoch + 1) % 20 == 0:
                if not args.no_early_stop and len(val_sequences) > 0:
                    def _bc_action_fn(s, d):
                        return bc_agent.select_action(s, d if bc_agent.drift_encoder is not None else np.zeros(1))
                    val_bc = evaluate_controller(val_sequences, dynamics_fn, _bc_action_fn, target_rr_norm)
                    if val_bc["mae"] < bc_best_val_mae:
                        bc_best_val_mae = val_bc["mae"]
                        bc_agent.save(bc_best_model_path)
                avg_loss = np.mean(epoch_bc_loss) if epoch_bc_loss else 0.0
                logger.info(f"BC Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")

        if not args.no_early_stop and os.path.exists(bc_best_model_path):
            logger.info(f"Loading best BC model (val MAE={bc_best_val_mae:.4f})")
            bc_agent.load(bc_best_model_path)
        bc_agent.save(bc_model_path)
        logger.info(f"CMP1 BC model saved to {bc_model_path}")

    # ============================================================
    # 6. Evaluate on Test Sequences
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 5: Offline Evaluation on CMP1 Test Sequences")
    logger.info("=" * 60)

    test_sequences = extract_test_sequences(test_data)
    if len(test_sequences) == 0:
        logger.warning("No test sequences found. Check terminal flags.")
        return

    evaluator = RolloutEvaluator(NORM_BOUNDS, rr_state_idx=RR_STATE_IDX)

    val_sequences = extract_test_sequences(val_data)
    dewma_params = tune_dewma(val_sequences, dynamics_fn, target_rr_norm, evaluator)

    controllers = {
        "SARC":          SARCController(agent),
        "SARC-no-drift": SARCController(no_drift_agent),
        "BC":            BCController(bc_agent),
        "D-EWMA":        DEWMAController(target_rr_norm, ACTION_DIM,
                                         lambda_0=dewma_params["lambda_0"],
                                         lambda_1=dewma_params["lambda_1"],
                                         action_bounds=NORM_BOUNDS),
        "Kalman":        KalmanR2RController(target_rr_norm, action_dim=ACTION_DIM, action_bounds=NORM_BOUNDS),
    }

    results_methods = {}
    for name, ctrl in controllers.items():
        logger.info(f"Evaluating {name}...")
        results_methods[name] = evaluator.evaluate(
            test_sequences, dynamics_fn, ctrl, target_rr_norm,
            noise_reset_fn=_reset_noise,
        )

    # ============================================================
    # 7. Print Results
    # ============================================================
    dynamics_label = "WorldModel(5-ensemble)"
    results = {
        "dataset": "CMP1 (public PHM)",
        "seed": args.seed,
        "n_test_sequences": len(test_sequences),
        "n_test_transitions": results_methods["SARC"]["n_transitions"],
        "dynamics_model": dynamics_label,
        "split": "chronological" if args.chrono_split else "random",
        "methods": results_methods,
    }

    stage_seqs = {
        "A": [s for s in test_sequences if s.get("stage", -1) == 0],
        "B": [s for s in test_sequences if s.get("stage", -1) == 1],
    }
    per_stage = {}
    if stage_seqs["A"] and stage_seqs["B"]:
        for stage_name, seqs in stage_seqs.items():
            stage_res = {}
            for name, ctrl in controllers.items():
                stage_res[name] = evaluator.evaluate(
                    seqs, dynamics_fn, ctrl, target_rr_norm, _reset_noise,
                )
            per_stage[stage_name] = stage_res
        results["per_stage"] = per_stage

    if args.cross_lot:
        split_tag = "Cross-Lot Sequences"
    elif args.chrono_split:
        split_tag = "Chrono Split"
    else:
        split_tag = "Random Split"
    print("\n" + "=" * 70)
    print(f"CMP1 Evaluation Results (WM Simulation, {split_tag})")
    print("=" * 70)
    print(f"{'Method':<20} {'MAE':>8} {'RMSE':>8} {'Act.Cost':>10} {'Spec Viol.':>12}")
    print("-" * 70)
    for name, m in results["methods"].items():
        sarc_improvement = ""
        if name != "SARC":
            imp = (m["mae"] - results_methods["SARC"]["mae"]) / m["mae"] * 100
            sarc_improvement = f"  [SARC -{imp:.1f}%]"
        print(f"{name:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['action_cost']:>10.4f} {m['spec_violation_rate']:>12.4f}{sarc_improvement}")
    print("=" * 70)

    if per_stage:
        for stage_name, stage_results in per_stage.items():
            print(f"\n-- Stage {stage_name} ({len(stage_seqs[stage_name])} sequences) --")
            print(f"{'Method':<20} {'MAE':>8} {'RMSE':>8} {'Act.Cost':>10}")
            print("-" * 50)
            sarc_mae = stage_results["SARC"]["mae"]
            for mname, m in stage_results.items():
                imp = ""
                if mname != "SARC":
                    pct = (m["mae"] - sarc_mae) / m["mae"] * 100
                    imp = f"  [SARC -{pct:.1f}%]"
                print(f"{mname:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['action_cost']:>10.4f}{imp}")

    print("\n(MAE/RMSE in normalized RR units; Action Cost = mean |Delta_a| per step)")

    results_path = os.path.join(RESULTS_DIR, f"cmp1_evaluation{suffix}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
