"""
CMP2 SARC Training + Evaluation Pipeline.

End-to-end: preprocess -> encoder pretrain -> CQL training -> Neural WM evaluation.
  - Val MAE best checkpoint selection
  - Neural World Model (5-ensemble) for dynamics rollout
  - Full baseline comparison (D-EWMA, Kalman, BC)

Usage:
  python src/rl/train_sarc.py                        # default ctx=2
  python src/rl/train_sarc.py --context-dim 4        # context dim sensitivity
  python src/rl/train_sarc.py --skip-train            # eval only (existing ckpt)

Paper final CMP2: cql_alpha=1.0, bc_weight=0.5, context_dim=2,
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

from src.data.config import RESULTS_DIR, CHECKPOINT_DIR
try:
    from src.data.preprocess_cmp2 import preprocess_cmp2
except ImportError:
    def preprocess_cmp2(*args, **kwargs):
        raise RuntimeError(
            "CMP2 dataset is private (industrial fab data) and not included "
            "in this submission package. To reproduce CMP2 results, use the "
            "pre-trained checkpoints with --skip-train and the provided result JSONs."
        )
from src.data.mdp_dataset import split_data
from src.rl.sarc_agent import SARCAgent
from src.rl.bc_agent import BCAgent
from src.baselines.dewma import DEWMAController
from src.baselines.kalman import KalmanR2RController
from src.evaluation.rollout_evaluator import (
    RolloutEvaluator, SARCController, BCController,
)

logger = logging.getLogger(__name__)

ACTION_DIM = 5
NORM_BOUNDS = (np.full(ACTION_DIM, -3.0), np.full(ACTION_DIM, 3.0))


# ── Sequence extraction ──────────────────────────────────────────────────
def extract_sequences(data: dict) -> list:
    """Split MDP dataset into per-lot sequences based on terminal flags."""
    terminals = data["terminals"]
    n = len(terminals)
    sequences, start = [], 0
    for i in range(n):
        if terminals[i] == 1.0 or i == n - 1:
            end = i + 1
            if end - start >= 2:
                sequences.append({k: v[start:end] for k, v in data.items()
                                  if isinstance(v, np.ndarray)})
            start = end
    return sequences


# ── Baseline tuning on validation set ───────────────────────────────────
# Industrial selection criterion: val_score = val_MAE + AC_WEIGHT * val_action_cost
AC_WEIGHT = 0.0


def tune_dewma(val_sequences, dynamics_fn, target_rr, evaluator) -> dict:
    """Grid search lambda_0 x lambda_1 on validation sequences (joint criterion)."""
    best_score, best_params = float("inf"), (0.5, 0.3)
    for l0 in [0.2, 0.3, 0.5]:
        for l1 in [0.5, 0.6, 0.7]:
            ctrl = DEWMAController(
                target_rr=target_rr, action_dim=ACTION_DIM,
                lambda_0=l0, lambda_1=l1, action_bounds=NORM_BOUNDS,
            )
            res = evaluator.evaluate(val_sequences, dynamics_fn, ctrl, target_rr)
            score = res["mae"] + AC_WEIGHT * res["action_cost"]
            if score < best_score:
                best_score = score
                best_params = (l0, l1)
    logger.info(f"D-EWMA tuned: lambda_0={best_params[0]}, lambda_1={best_params[1]} "
                f"(val score={best_score:.4f})")
    return {"lambda_0": best_params[0], "lambda_1": best_params[1]}


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SARC on CMP2")
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cql-alpha", type=float, default=1.0,
                        help="Paper final CMP2: alpha_CQL=1.0 (via run_cmp2_final.py).")
    parser.add_argument("--context-dim", type=int, default=2)
    parser.add_argument("--bc-weight", type=float, default=0.5,
                        help="TD3+BC weight. 0=pure CQL actor loss.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Checkpoint/result suffix. Default: ctx{context_dim}_s{seed}")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, evaluate existing checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--lambda-s", type=float, default=0.01,
                        help="Action cost coefficient lambda_s (0=tracking-only reward)")
    parser.add_argument("--use-wm", action="store_true",
                        help="[deprecated -- Neural World Model is always used]")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_name is None:
        args.run_name = f"ctx{args.context_dim}_s{args.seed}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ================================================================
    # 1. Data
    # ================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Data Preprocessing")
    logger.info("=" * 60)

    from src.data.config import RewardConfig
    rc = RewardConfig(beta=args.lambda_s)
    mdp_data, targets, s_scaler, a_scaler = preprocess_cmp2(reward_config=rc)
    train_data, val_data, test_data = split_data(mdp_data)

    # Re-seed after split_data (which internally resets to seed=42)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = train_data["observations"].shape[1]
    action_dim = train_data["actions"].shape[1]
    drift_dim = train_data["drift_features"].shape[1]
    rr_start = state_dim - 4  # last 4 dims = AKE3, AK, AKE1, AKE2 (prev RR)

    target_rr = float(np.mean(train_data["observations"][:, rr_start:]))
    rr_std_raw = np.mean([t["std"] for t in targets.values()])
    spec_margin = 450.0 / rr_std_raw  # AKE3 spec: 2850 +/- 450

    logger.info(f"State: {state_dim}D, Action: {action_dim}D, Drift: {drift_dim}D")
    logger.info(f"Train: {len(train_data['observations'])}, "
                f"Val: {len(val_data['observations'])}, "
                f"Test: {len(test_data['observations'])}")
    logger.info(f"Target RR (norm): {target_rr:.4f}, Spec margin (norm): {spec_margin:.4f}")

    # ================================================================
    # 2. Neural World Model (5-ensemble dynamics proxy)
    # ================================================================
    logger.info("=" * 60)
    logger.info("Step 2: Load Neural World Model")
    logger.info("=" * 60)

    from src.models.world_model import WorldModel
    wm_ckpt = os.path.join(CHECKPOINT_DIR, "world_model_cmp2_best.pt")
    if not os.path.exists(wm_ckpt):
        raise FileNotFoundError(
            f"CMP2 World Model not found at {wm_ckpt}. "
            "Run: python src/models/train_world_model.py first."
        )
    eval_wm = WorldModel(state_dim=state_dim, action_dim=action_dim, device="cpu")
    eval_wm._load_checkpoint("cmp2_best")
    eval_wm.ensemble.eval()
    logger.info(f"Using Neural World Model for evaluation: {wm_ckpt}")

    def dynamics_fn(state, action):
        """Predict mean RR from Neural WM (5-ensemble, delta prediction)."""
        next_s, _, _ = eval_wm.predict(
            state.reshape(1, -1).astype(np.float32),
            action.reshape(1, -1).astype(np.float32),
            deterministic=True,
        )
        return float(next_s[0, rr_start:rr_start + 4].mean())

    val_sequences = extract_sequences(val_data)
    test_sequences = extract_sequences(test_data)
    logger.info(f"Val sequences: {len(val_sequences)}, Test sequences: {len(test_sequences)}")

    evaluator = RolloutEvaluator(
        NORM_BOUNDS, rr_state_idx=rr_start, rr_state_end=rr_start + 4,
        spec_margin_norm=spec_margin,
    )

    # ================================================================
    # 3. SARC Training
    # ================================================================
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"sarc_model_{args.run_name}.pt")
    best_path = ckpt_path.replace(".pt", "_best.pt")
    nodrift_ckpt = os.path.join(CHECKPOINT_DIR, f"sarc_no_drift_{args.run_name}.pt")
    nodrift_best = nodrift_ckpt.replace(".pt", "_best.pt")
    bc_ckpt = os.path.join(CHECKPOINT_DIR, f"bc_model_{args.run_name}.pt")
    bc_best = bc_ckpt.replace(".pt", "_best.pt")

    device = args.device
    if device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available()
                  else "cpu")

    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("Step 3: SARC Training")
        logger.info("=" * 60)

        agent = SARCAgent(
            state_dim=state_dim, action_dim=action_dim,
            drift_dim=drift_dim, context_dim=args.context_dim,
            lr_actor=args.lr, lr_critic=args.lr,
            cql_alpha=args.cql_alpha, bc_weight=args.bc_weight,
            device=device,
        )

        # 3a: Drift Encoder pretrain
        logger.info(f"--- Drift Encoder pretrain ({args.pretrain_epochs} epochs) ---")
        train_drift = torch.FloatTensor(train_data["drift_features"])
        train_rr = torch.FloatTensor(train_data["observations"][:, rr_start:])
        losses = agent.pretrain_encoder(
            train_drift, train_rr,
            n_epochs=args.pretrain_epochs, batch_size=args.batch_size,
        )
        logger.info(f"Encoder pretrain: {losses[0]:.4f} -> {losses[-1]:.4f}")

        # 3b: CQL Training with val MAE best checkpoint
        logger.info(f"--- CQL Training ({args.epochs} epochs) ---")
        dev = agent.device
        states = torch.FloatTensor(train_data["observations"]).to(dev)
        actions = torch.FloatTensor(train_data["actions"]).to(dev)
        rewards = torch.FloatTensor(train_data["rewards"]).unsqueeze(-1).to(dev)
        next_states = torch.FloatTensor(train_data["next_observations"]).to(dev)
        terminals = torch.FloatTensor(train_data["terminals"]).unsqueeze(-1).to(dev)
        drift_t = torch.FloatTensor(train_data["drift_features"]).to(dev)
        next_drift_t = torch.FloatTensor(train_data["next_drift_features"]).to(dev)
        n_samples = len(states)

        best_val_mae = float("inf")

        for epoch in range(args.epochs):
            agent.actor.train()
            agent.drift_encoder.train()
            idx = np.random.permutation(n_samples)
            loss_log = {"critic_loss": [], "actor_loss": [], "cql_loss": []}

            for start in range(0, n_samples, args.batch_size):
                b = idx[start:start + args.batch_size]
                if len(b) < 8:
                    continue
                L = agent.update(
                    states[b], actions[b], rewards[b],
                    next_states[b], terminals[b], drift_t[b], next_drift_t[b],
                )
                for k in loss_log:
                    if k in L:
                        loss_log[k].append(L[k])

            if (epoch + 1) % 20 == 0:
                agent.actor.eval()
                agent.drift_encoder.eval()
                vm = evaluator.evaluate(
                    val_sequences, dynamics_fn, SARCController(agent), target_rr,
                )
                if vm["mae"] < best_val_mae:
                    best_val_mae = vm["mae"]
                    agent.save(best_path)

                crit = np.mean(loss_log["critic_loss"]) if loss_log["critic_loss"] else 0
                act = np.mean(loss_log["actor_loss"]) if loss_log["actor_loss"] else 0
                cql = np.mean(loss_log["cql_loss"]) if loss_log["cql_loss"] else 0
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Critic: {crit:.4f} | Actor: {act:.4f} | CQL: {cql:.4f} | "
                    f"ValMAE: {vm['mae']:.4f}"
                )

        if os.path.exists(best_path):
            logger.info(f"Loading best SARC (val MAE={best_val_mae:.4f})")
            agent.load(best_path)
        agent.save(ckpt_path)

        # 3c: SARC-no-drift ablation
        logger.info(f"--- SARC-no-drift ablation ({args.epochs} epochs) ---")
        nodrift_agent = SARCAgent(
            state_dim=state_dim, action_dim=action_dim,
            drift_dim=drift_dim, context_dim=args.context_dim,
            lr_actor=args.lr, lr_critic=args.lr,
            cql_alpha=args.cql_alpha, bc_weight=args.bc_weight,
            use_drift=False, device=device,
        )

        best_nd_mae = float("inf")
        for epoch in range(args.epochs):
            nodrift_agent.actor.train()
            idx = np.random.permutation(n_samples)
            for start in range(0, n_samples, args.batch_size):
                b = idx[start:start + args.batch_size]
                if len(b) < 8:
                    continue
                nodrift_agent.update(
                    states[b], actions[b], rewards[b],
                    next_states[b], terminals[b], drift_t[b], next_drift_t[b],
                )
            if (epoch + 1) % 20 == 0:
                nodrift_agent.actor.eval()
                nd_res = evaluator.evaluate(
                    val_sequences, dynamics_fn, SARCController(nodrift_agent), target_rr,
                )
                if nd_res["mae"] < best_nd_mae:
                    best_nd_mae = nd_res["mae"]
                    nodrift_agent.save(nodrift_best)
                if (epoch + 1) % 40 == 0:
                    logger.info(f"  [no-drift] Epoch {epoch+1} | ValMAE={nd_res['mae']:.4f}")

        if os.path.exists(nodrift_best):
            nodrift_agent.load(nodrift_best)
        nodrift_agent.save(nodrift_ckpt)

        # 3d: BC (Behavioral Cloning)
        logger.info(f"--- BC Training ({args.epochs} epochs) ---")
        bc_agent = BCAgent(
            state_dim=state_dim, action_dim=action_dim,
            drift_dim=drift_dim, context_dim=args.context_dim,
            device=device,
        )
        bc_agent.pretrain_encoder(
            train_drift, train_rr,
            n_epochs=args.pretrain_epochs, batch_size=args.batch_size,
        )

        best_bc_mae = float("inf")
        for epoch in range(args.epochs):
            bc_agent.actor.train()
            if bc_agent.drift_encoder is not None:
                bc_agent.drift_encoder.train()
            idx = np.random.permutation(n_samples)
            for start in range(0, n_samples, args.batch_size):
                b = idx[start:start + args.batch_size]
                if len(b) < 8:
                    continue
                bc_agent.train_step(states[b], actions[b], drift_t[b])

            if (epoch + 1) % 20 == 0:
                bc_agent.actor.eval()
                if bc_agent.drift_encoder is not None:
                    bc_agent.drift_encoder.eval()
                bc_res = evaluator.evaluate(
                    val_sequences, dynamics_fn, BCController(bc_agent), target_rr,
                )
                if bc_res["mae"] < best_bc_mae:
                    best_bc_mae = bc_res["mae"]
                    bc_agent.save(bc_best)
                if (epoch + 1) % 40 == 0:
                    logger.info(f"  [BC] Epoch {epoch+1} | ValMAE={bc_res['mae']:.4f}")

        if os.path.exists(bc_best):
            bc_agent.load(bc_best)
        bc_agent.save(bc_ckpt)

    else:
        # --skip-train: load existing checkpoints
        logger.info("Step 3: Skipping training (--skip-train)")
        agent = SARCAgent(
            state_dim=state_dim, action_dim=action_dim,
            drift_dim=drift_dim, context_dim=args.context_dim, device=device,
        )
        _sarc_load = best_path if os.path.exists(best_path) else ckpt_path
        if os.path.exists(_sarc_load):
            agent.load(_sarc_load)
            logger.info(f"Loaded: {_sarc_load}")
        else:
            logger.warning(f"Checkpoint not found: {best_path}")

        nodrift_agent = SARCAgent(
            state_dim=state_dim, action_dim=action_dim,
            drift_dim=drift_dim, context_dim=args.context_dim,
            use_drift=False, device=device,
        )
        _nd_load = nodrift_best if os.path.exists(nodrift_best) else nodrift_ckpt
        if os.path.exists(_nd_load):
            nodrift_agent.load(_nd_load)
            logger.info(f"Loaded: {_nd_load}")
        else:
            logger.warning(f"No-drift checkpoint not found: {nodrift_best}")

        bc_agent = BCAgent(
            state_dim=state_dim, action_dim=action_dim,
            drift_dim=drift_dim, context_dim=args.context_dim, device=device,
        )
        _bc_load = bc_best if os.path.exists(bc_best) else bc_ckpt
        if os.path.exists(_bc_load):
            bc_agent.load(_bc_load)
            logger.info(f"Loaded: {_bc_load}")
        else:
            logger.warning(f"BC checkpoint not found: {bc_best}")

    # ================================================================
    # 4. Evaluation
    # ================================================================
    logger.info("=" * 60)
    logger.info("Step 4: Evaluation (Neural WM, sequence-level)")
    logger.info("=" * 60)

    dewma_params = tune_dewma(val_sequences, dynamics_fn, target_rr, evaluator)

    controllers = {
        "SARC":          SARCController(agent),
        "SARC-no-drift": SARCController(nodrift_agent),
        "BC":            BCController(bc_agent),
        "D-EWMA":        DEWMAController(
                             target_rr=target_rr, action_dim=action_dim,
                             lambda_0=dewma_params["lambda_0"],
                             lambda_1=dewma_params["lambda_1"],
                             action_bounds=NORM_BOUNDS),
        "Kalman":        KalmanR2RController(
                             target_rr=target_rr, action_dim=action_dim,
                             action_bounds=NORM_BOUNDS),
    }

    methods = {}
    for name, ctrl in controllers.items():
        logger.info(f"Evaluating: {name}")
        methods[name] = evaluator.evaluate(test_sequences, dynamics_fn, ctrl, target_rr)

    # ================================================================
    # 5. Results
    # ================================================================
    print(f"\n{'='*80}")
    print(f"CMP2 Results  [run={args.run_name}  ctx_dim={args.context_dim}]")
    print(f"Target RR (norm): {target_rr:.4f} | Spec margin (norm): {spec_margin:.4f}")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'MAE':>8} {'RMSE':>8} {'Act.Cost':>10} {'SpecViol':>10}")
    print("-" * 60)

    sarc_mae = methods["SARC"]["mae"]
    for name, m in methods.items():
        imp = (f"  [vs SARC +{(m['mae'] - sarc_mae) / sarc_mae * 100:.1f}%]"
               if name != "SARC" and sarc_mae > 0 else "")
        print(f"{name:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['action_cost']:>10.4f} {m['spec_violation_rate']:>10.4f}{imp}")

    results = {
        "dataset": "CMP2 (private fab data)",
        "run_name": args.run_name,
        "context_dim": args.context_dim,
        "n_test_sequences": len(test_sequences),
        "dynamics_model": "Neural WM (5-ensemble)",
        "seed": args.seed,
        "methods": methods,
    }
    results_path = os.path.join(RESULTS_DIR, f"sarc_evaluation_{args.run_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
