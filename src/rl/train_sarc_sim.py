"""
Preston CMP Simulator-based SARC Training and Evaluation.

Follows the same structure as train_sarc_cmp1.py:
  - Evaluation: PrestonRolloutEvaluator -- CMPSimulator ground-truth rollout
  - STATE_DIM=7, ACTION_DIM=6, DRIFT_DIM=9
  - drift_scale argument supports mild/medium/heavy conditions
  - Results saved to: results/sim_evaluation_{tag}.json

Usage:
  python src/rl/train_sarc_sim.py --drift-scale 1.0   # mild
  python src/rl/train_sarc_sim.py --drift-scale 2.0   # medium

Paper final Sim (mild/medium): cql_alpha=1.0, bc_weight=0.5, context_dim=2,
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
from src.data.cmp_simulator import SimConfig
from src.data.generate_sim_dataset import generate_dataset, normalize_and_split
from src.data.mdp_dataset import save_dataset
from src.rl.sarc_agent import SARCAgent
from src.rl.bc_agent import BCAgent
from src.baselines.dewma import DEWMAController
from src.baselines.kalman import KalmanR2RController
from src.evaluation.rollout_evaluator import (
    SARCController, BCController,
)
from src.evaluation.preston_rollout import PrestonRolloutEvaluator

logger = logging.getLogger(__name__)

N_ZONES      = 5
ACTION_DIM   = N_ZONES + 1        # (P_1..P_5, V)
STATE_DIM    = 1 + N_ZONES + 1    # (prev_RR, prev_P_1..P_5, prev_V)
DRIFT_DIM    = 9                  # wear(4) + delta_wear(4) + lot_position(1)
RR_OUT_DIM   = 1
RR_STATE_IDX = 0
ACTION_STATE_INDICES = tuple(range(1, 1 + ACTION_DIM))  # prev-action block in state
RUNS_PER_LOT = 20

NORM_BOUNDS = (np.full(ACTION_DIM, -3.0), np.full(ACTION_DIM, 3.0))


# ── Sequence extraction ───────────────────────────────────────────────────
def extract_test_sequences(test_data: dict) -> list:
    terminals = test_data["terminals"]
    n = len(terminals)
    sequences, seq_start = [], 0
    for i in range(n):
        if terminals[i] == 1.0 or i == n - 1:
            seq_end = i + 1
            if seq_end - seq_start >= 2:
                sequences.append({k: v[seq_start:seq_end] for k, v in test_data.items()})
            seq_start = seq_end
    logger.info(
        f"Extracted {len(sequences)} test sequences "
        f"(avg len: {np.mean([len(s['observations']) for s in sequences]):.1f})"
    )
    return sequences


# ── Baseline tuning on validation set (joint MAE + action cost) ──────────
AC_WEIGHT = 0.0  # industrial-scale action cost penalty; see train_sarc.py for rationale


def tune_dewma(val_sequences, evaluator: PrestonRolloutEvaluator, target_rr_norm=0.0) -> dict:
    """Grid search lambda_0 x lambda_1 on validation sequences (joint criterion)."""
    best_score, best_params = float("inf"), (0.5, 0.3)
    logger.info("Tuning D-EWMA lambdas on validation set ...")
    for l0 in [0.2, 0.3, 0.5]:
        for l1 in [0.5, 0.6, 0.7]:
            ctrl = DEWMAController(
                target_rr=target_rr_norm, action_dim=ACTION_DIM,
                lambda_0=l0, lambda_1=l1, action_bounds=NORM_BOUNDS,
            )
            res = evaluator.evaluate(val_sequences, ctrl, target_rr_norm)
            score = res["mae"] + AC_WEIGHT * res["action_cost"]
            if score < best_score:
                best_score = score
                best_params = (l0, l1)
    logger.info(
        f"Best D-EWMA: lambda_0={best_params[0]}, lambda_1={best_params[1]} "
        f"(val score={best_score:.4f})"
    )
    return {"lambda_0": best_params[0], "lambda_1": best_params[1]}


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SARC on CMP simulator")
    parser.add_argument("--drift-scale", type=float, default=1.0,
                        help="Drift intensity: mild=1.0, medium=2.0")
    parser.add_argument("--tag", type=str, default="",
                        help="Result file suffix (auto-set from drift-scale if empty)")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--pretrain-epochs", type=int,   default=80)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--cql-alpha",       type=float, default=1.0,
                        help="Paper final Sim: alpha_CQL=1.0 (mild/medium common).")
    parser.add_argument("--context-dim",     type=int,   default=2)
    parser.add_argument("--bc-weight",       type=float, default=0.5)
    parser.add_argument("--device",          type=str,   default="auto")
    parser.add_argument("--lambda-s",        type=float, default=0.01,
                        help="Action cost coefficient lambda_s (0=tracking-only reward)")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-datagen",    action="store_true",
                        help="Skip data generation, load existing files")
    parser.add_argument("--skip-train",      action="store_true",
                        help="Skip SARC training, load existing checkpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # auto-tag: base tag for data files (seed-independent)
    if not args.tag:
        if args.drift_scale <= 0.5:  args.tag = "sim_mild"
        else:                        args.tag = "sim_medium"
    args.tag = f"{args.tag}_cql{args.cql_alpha}"  # include cql_alpha in tag (matches checkpoint naming)
    if args.lambda_s != 0.01:
        args.tag = f"{args.tag}_ls{args.lambda_s}"
    base_tag = args.tag
    tag = f"{base_tag}_s{args.seed}" if args.seed != 42 else base_tag

    # ── 1. Data generation ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Step 1: Dataset  tag={tag}  drift_scale={args.drift_scale}")
    logger.info("=" * 60)

    train_path = os.path.join(RESULTS_DIR, "data", f"{base_tag}_train_observations.npy")
    if os.path.exists(train_path):
        logger.info(f"Loading existing dataset: {base_tag}")
        from src.data.mdp_dataset import load_dataset
        data_dir = os.path.join(RESULTS_DIR, "data")
        def _load(split):
            d, meta = load_dataset(f"{base_tag}_{split}")
            for key in ["drift_features", "next_drift_features", "stage_labels", "file_ids"]:
                p = os.path.join(data_dir, f"{base_tag}_{split}_{key}.npy")
                if os.path.exists(p):
                    d[key] = np.load(p)
            return d, meta
        train_data, train_meta = _load("train")
        val_data,   _          = _load("val")
        test_data,  _          = _load("test")
        scalers = train_meta.get("scalers", {})
        if not scalers or "state" not in scalers:
            raise RuntimeError(
                f"Scalers missing from {base_tag}_train_metadata.pkl; "
                f"regenerate data by removing {data_dir}/{base_tag}_* files."
            )
    else:
        mdp_raw = generate_dataset(
            n_lots=400, runs_per_lot=RUNS_PER_LOT,
            drift_scale=args.drift_scale, seed=42, tag=base_tag,
            lambda_s=args.lambda_s,
        )
        train_data, val_data, test_data, scalers = normalize_and_split(mdp_raw, tag=base_tag)
        save_dataset(train_data, scalers,      {"target_rr": 0.0}, f"{base_tag}_train")
        save_dataset(val_data,   {},           {"target_rr": 0.0}, f"{base_tag}_val")
        save_dataset(test_data,  {},           {"target_rr": 0.0}, f"{base_tag}_test")

    # Re-seed after data loading for training stochasticity only
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(
        f"Train: {len(train_data['observations'])}, "
        f"Val: {len(val_data['observations'])}, "
        f"Test: {len(test_data['observations'])}"
    )

    # ── 2. Preston ground-truth evaluator ─────────────────────────────────
    logger.info("Step 2: Instantiate Preston ground-truth evaluator")
    sim_cfg = SimConfig(drift_scale=args.drift_scale, seed=42)
    preston_evaluator = PrestonRolloutEvaluator(
        cfg=sim_cfg,
        scalers=scalers,
        runs_per_lot=RUNS_PER_LOT,
        action_bounds=NORM_BOUNDS,
        rr_state_idx=RR_STATE_IDX,
        action_state_indices=ACTION_STATE_INDICES,
    )

    target_rr_norm = 0.0  # target = 0 in normalized space

    # ── 3. SARC Agent ──────────────────────────────────────────────────
    agent = SARCAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        drift_dim=DRIFT_DIM,
        context_dim=args.context_dim,
        lr_actor=args.lr,
        lr_critic=args.lr,
        cql_alpha=args.cql_alpha,
        rr_out_dim=RR_OUT_DIM,
        bc_weight=args.bc_weight,
        device=args.device,
    )

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"sarc_{tag}_model.pt")
    best_path = ckpt_path.replace(".pt", "_best.pt")

    val_sequences = extract_test_sequences(val_data)

    no_drift_agent = SARCAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, drift_dim=DRIFT_DIM,
        context_dim=args.context_dim, lr_actor=args.lr, lr_critic=args.lr,
        cql_alpha=args.cql_alpha, rr_out_dim=RR_OUT_DIM,
        bc_weight=args.bc_weight, use_drift=False, device=args.device,
    )

    no_drift_ckpt = ckpt_path.replace(".pt", "_nodrift.pt")
    no_drift_best = ckpt_path.replace(".pt", "_nodrift_best.pt")
    bc_ckpt = ckpt_path.replace(".pt", "_bc.pt")
    bc_best = ckpt_path.replace(".pt", "_bc_best.pt")

    if args.skip_train and (os.path.exists(ckpt_path) or os.path.exists(best_path)):
        _load = best_path if os.path.exists(best_path) else ckpt_path
        logger.info(f"Loading existing model: {_load}")
        agent.load(_load)

        _nd_load = no_drift_best if os.path.exists(no_drift_best) else no_drift_ckpt
        if os.path.exists(_nd_load):
            no_drift_agent.load(_nd_load)
        else:
            logger.warning(f"No-drift checkpoint not found: {no_drift_best}. Using random init.")

        bc_agent = BCAgent(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            drift_dim=DRIFT_DIM, context_dim=args.context_dim,
            rr_out_dim=RR_OUT_DIM, device=args.device,
        )
        _bc_load = bc_best if os.path.exists(bc_best) else bc_ckpt
        if os.path.exists(_bc_load):
            bc_agent.load(_bc_load)
        else:
            logger.warning(f"BC checkpoint not found: {bc_best}. Using random init.")

    else:
        # ── 4. DriftEncoder Pretrain ──────────────────────────────────────
        logger.info("Step 3: Pretrain DriftEncoder")
        train_drift = torch.FloatTensor(train_data["drift_features"])
        train_rr    = torch.FloatTensor(
            train_data["observations"][:, RR_STATE_IDX:RR_STATE_IDX + 1]
        )
        losses = agent.pretrain_encoder(
            train_drift, train_rr,
            n_epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
        )
        logger.info(f"Encoder pretrain done. Final loss: {losses[-1]:.6f}")

        # ── 5. CQL Training ───────────────────────────────────────────────
        logger.info("Step 4: CQL Training")
        dev = agent.device
        states      = torch.FloatTensor(train_data["observations"]).to(dev)
        actions     = torch.FloatTensor(train_data["actions"]).to(dev)
        rewards     = torch.FloatTensor(train_data["rewards"]).unsqueeze(-1).to(dev)
        next_states = torch.FloatTensor(train_data["next_observations"]).to(dev)
        terminals   = torch.FloatTensor(train_data["terminals"]).unsqueeze(-1).to(dev)
        drift       = torch.FloatTensor(train_data["drift_features"]).to(dev)
        next_drift  = torch.FloatTensor(train_data["next_drift_features"]).to(dev)
        n_samples = len(states)

        val_sequences = extract_test_sequences(val_data)
        best_val_mae  = float("inf")

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
                    next_states[b], terminals[b], drift[b], next_drift[b],
                )
                for k in loss_log:
                    if k in L:
                        loss_log[k].append(L[k])

            if (epoch + 1) % 20 == 0:
                agent.actor.eval()
                agent.drift_encoder.eval()
                vm = preston_evaluator.evaluate(val_sequences, SARCController(agent), target_rr_norm)
                if vm["mae"] < best_val_mae:
                    best_val_mae = vm["mae"]
                    agent.save(best_path)
                crit = np.mean(loss_log["critic_loss"]) if loss_log["critic_loss"] else 0
                act  = np.mean(loss_log["actor_loss"])  if loss_log["actor_loss"]  else 0
                cql  = np.mean(loss_log["cql_loss"])    if loss_log["cql_loss"]    else 0
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Critic: {crit:.4f} | Actor: {act:.4f} | CQL: {cql:.4f} | "
                    f"ValMAE: {vm['mae']:.4f}"
                )

        if os.path.exists(best_path):
            logger.info(f"Loading best model (val MAE={best_val_mae:.4f})")
            agent.load(best_path)
        agent.save(ckpt_path)
        logger.info(f"Model saved: {ckpt_path}")

        # ── SARC-no-drift ablation (use_drift=False) ──────────────────────
        logger.info("Step 5b: Train SARC-no-drift ablation (use_drift=False)")
        no_drift_agent = SARCAgent(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, drift_dim=DRIFT_DIM,
            context_dim=args.context_dim, lr_actor=args.lr, lr_critic=args.lr,
            cql_alpha=args.cql_alpha, rr_out_dim=RR_OUT_DIM,
            bc_weight=args.bc_weight, use_drift=False, device=args.device,
        )
        no_drift_ckpt = ckpt_path.replace(".pt", "_nodrift.pt")
        no_drift_best = no_drift_ckpt.replace(".pt", "_best.pt")

        best_nd_mae = float("inf")
        for epoch in range(args.epochs):
            no_drift_agent.actor.train()
            idx = np.random.permutation(n_samples)
            for start in range(0, n_samples, args.batch_size):
                b = idx[start:start + args.batch_size]
                if len(b) < 8:
                    continue
                no_drift_agent.update(
                    states[b], actions[b], rewards[b],
                    next_states[b], terminals[b], drift[b], next_drift[b],
                )
            if (epoch + 1) % 20 == 0:
                no_drift_agent.actor.eval()
                nd_res = preston_evaluator.evaluate(val_sequences, SARCController(no_drift_agent), target_rr_norm)
                if nd_res["mae"] < best_nd_mae:
                    best_nd_mae = nd_res["mae"]
                    no_drift_agent.save(no_drift_best)
                if (epoch + 1) % 40 == 0:
                    logger.info(f"  [no-drift] Epoch {epoch+1} | ValMAE={nd_res['mae']:.4f}")
        if os.path.exists(no_drift_best):
            no_drift_agent.load(no_drift_best)
        no_drift_agent.save(no_drift_ckpt)

        # ── BC (Behavioral Cloning) training ──────────────────────────────
        logger.info("Step 5c: Train BC agent (pure imitation learning)")
        bc_agent = BCAgent(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            drift_dim=DRIFT_DIM, context_dim=args.context_dim,
            rr_out_dim=RR_OUT_DIM, device=args.device,
        )
        bc_ckpt = ckpt_path.replace(".pt", "_bc.pt")
        bc_best = bc_ckpt.replace(".pt", "_best.pt")
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
                bc_agent.train_step(states[b], actions[b], drift[b])
            if (epoch + 1) % 20 == 0:
                bc_agent.actor.eval()
                if bc_agent.drift_encoder is not None:
                    bc_agent.drift_encoder.eval()
                bc_res = preston_evaluator.evaluate(val_sequences, BCController(bc_agent), target_rr_norm)
                if bc_res["mae"] < best_bc_mae:
                    best_bc_mae = bc_res["mae"]
                    bc_agent.save(bc_best)
                if (epoch + 1) % 40 == 0:
                    logger.info(f"  [BC] Epoch {epoch+1} | ValMAE={bc_res['mae']:.4f}")
        if os.path.exists(bc_best):
            bc_agent.load(bc_best)
        bc_agent.save(bc_ckpt)

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    logger.info("Step 5: Evaluate all methods on test sequences (Preston ground truth)")
    test_sequences = extract_test_sequences(test_data)
    dewma_params = tune_dewma(val_sequences, preston_evaluator, target_rr_norm)
    logger.info(f"D-EWMA tuned: lambda_0={dewma_params['lambda_0']}, lambda_1={dewma_params['lambda_1']}")

    controllers = {
        "SARC":          SARCController(agent),
        "SARC-no-drift": SARCController(no_drift_agent),
        "BC":            BCController(bc_agent),
        "D-EWMA":        DEWMAController(
                             target_rr=target_rr_norm, action_dim=ACTION_DIM,
                             lambda_0=dewma_params["lambda_0"],
                             lambda_1=dewma_params["lambda_1"],
                             action_bounds=NORM_BOUNDS),
        "Kalman":        KalmanR2RController(target_rr=target_rr_norm, action_dim=ACTION_DIM,
                             action_bounds=NORM_BOUNDS),
    }

    methods = {}
    for name, ctrl in controllers.items():
        logger.info(f"Evaluating {name}...")
        methods[name] = preston_evaluator.evaluate(test_sequences, ctrl, target_rr_norm)

    # ── Output ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Simulation Results  [{tag}  drift_scale={args.drift_scale}]")
    print("=" * 70)
    print(f"{'Method':<20} {'MAE':>8} {'RMSE':>8} {'Act.Cost':>10} {'SpecViol':>10}")
    print("-" * 60)
    sarc_mae = methods["SARC"]["mae"]
    for name, m in methods.items():
        imp = (
            f"  [SARC -{(m['mae'] - sarc_mae) / m['mae'] * 100:.1f}%]"
            if name != "SARC" else ""
        )
        print(
            f"{name:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
            f"{m['action_cost']:>10.4f} {m['spec_violation_rate']:>10.4f}{imp}"
        )
    print("=" * 70)

    results = {
        "dataset": f"CMP Simulator (Preston eq., drift_scale={args.drift_scale})",
        "tag": tag,
        "seed": args.seed,
        "n_test_sequences": len(test_sequences),
        "dynamics_model": "CMPSimulator (Preston eq. ground truth)",
        "methods": methods,
    }
    results_path = os.path.join(RESULTS_DIR, f"sim_evaluation_{tag}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")
    return results


if __name__ == "__main__":
    main()
