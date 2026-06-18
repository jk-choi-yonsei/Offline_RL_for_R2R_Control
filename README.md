# SARC — Drift-Aware Offline RL for Run-to-Run Control in CMP

Reproducibility package for the IEEE *Transactions on Semiconductor Manufacturing*
paper **"Drift-Aware Offline Reinforcement Learning for Run-to-Run Control in
Chemical Mechanical Planarization"** (manuscript TSM-26-0126).

**SARC** = a Drift Encoder (removal-rate–prediction pretraining, 2-D context
bottleneck) feeding a drift-conditioned Conservative Q-Learning (CQL)
actor–critic. It is compared against BC, double-EWMA (standard and
consumable-age variants), and a Kalman R2R controller, on a CMP drift
simulator, the public PHM 2016 benchmark (CMP1), and an industrial fab
dataset (CMP2).

## Layout
```
src/
  rl/          sarc_agent, bc_agent, reward, train_sarc{,_cmp1,_sim}
  models/      drift_encoder, world_model (5-ensemble), train_world_model{,_cmp1}, dynamics_mlp
  baselines/   dewma_drift (DoubleEWMA + AgeBasedDoubleEWMA), kalman
  data/        config, preprocess_cmp1_r2r, preprocess_cmp2, cmp_simulator,
               generate_sim_dataset, mdp_dataset
  evaluation/  rollout_evaluator, preston_rollout, noise_models
scripts/       verify_reproduction, sweep_gamma, wilcoxon_holm, aggregate_5seed
results/
  checkpoints/ SARC / SARC-no-drift / BC / raw-wear (5 seeds) + world models
  *.json       evaluation grids, ablations, statistics (results of record)
```

## Install
```
pip install -r requirements.txt        # Python 3.10+
```

## Datasets
- **Simulator** (Preston equation) — generated locally:
  `python -m src.data.generate_sim_dataset`  (drift_scale 0.4 = mild, 1.0 = medium)
- **CMP1** (public PHM 2016 challenge) — bundled under `Dataset/CMP1/`
  (good-faith mirror, since the original PHM download is no longer available; see
  `Dataset/CMP1/LICENSE.txt`), then `python -m src.data.preprocess_cmp1_r2r`
- **CMP2** (industrial fab data) — **proprietary; not redistributable.**
  Trained checkpoints and evaluation JSONs are included for inspection, but the
  raw CMP2 data cannot be shared, so CMP2 cannot be retrained from this package.

## Reproduce the reported numbers (read-only, no retraining)
```
python scripts/verify_reproduction.py     # checks shipped JSONs vs paper (SARC point values,
                                          #   dominance, OPE, age-D-EWMA, NU, matched-box significance)
python scripts/sweep_gamma.py --gammas 0,0.05,0.1,0.3,1.0 --box 1.0   # MAE-CE grid / Pareto front
python scripts/wilcoxon_holm.py           # per-sequence Holm-corrected significance (learned policies)
python scripts/aggregate_5seed.py         # 5-seed mean +/- std summary
```
`verify_reproduction.py` should print **ALL CHECKS PASS**.

## Retrain (simulator / CMP1)
```
# Simulator uses Preston ground-truth rollouts (no world model needed):
python -m src.rl.train_sarc_sim  --drift-scale 0.4    # mild
python -m src.rl.train_sarc_sim  --drift-scale 1.0    # medium

# CMP1 uses a neural world model; train it first, then SARC:
python -m src.models.train_world_model_cmp1
python -m src.rl.train_sarc_cmp1                       # see --help for the chronological split
```
Each training run produces SARC, SARC-no-drift, and BC checkpoints + a
per-seed evaluation JSON. Paper seeds: **123, 789, 1024, 2024, 7777**.

## Notes
- **Baselines are deterministic** (no checkpoints): double-EWMA (Butler &
  Stefani; Chen consumable-age variant) and Kalman. They are tuned on the
  validation set by `val MAE + 0.1 * CE`; `sweep_gamma.py` shows the comparison
  is insensitive to this weight.
- All controllers are evaluated under a **matched ±1σ action box**.
- **MAE** is normalized per dataset; **CE** (control effort) is the per-zone
  z-scored recipe change. CMP1/CMP2 are evaluated with the 5-ensemble neural
  world model; the simulator uses the Preston ground truth.
