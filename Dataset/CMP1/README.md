# CMP1 — PHM 2016 Data Challenge (Mirror)

This directory bundles the **PHM 2016 Data Challenge — CMP Dataset** for academic reproducibility of the SARC paper. See `LICENSE.txt` for redistribution terms and the takedown clause.

## Provenance

- Original source: PHM Society, *PHM 2016 Data Challenge — CMP Dataset* (2016).
- Original page: <https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/>
- This mirror is provided in good faith because the original ZIP downloads on the PHM page are no longer functional. PHM Society retains all rights; see `LICENSE.txt`.

## Directory layout

```
Dataset/CMP1/
├── LICENSE.txt                       # Redistribution rationale + takedown clause
├── README.md                         # This file
├── CMP-training-removalrate.csv      # Wafer-level RR labels
└── CMP-data/
    └── training/                     # 185 per-wafer trace CSVs (≈130 MB)
        └── CMP-training-{000..184}.csv
```

The PHM-issued `CMP-data/test/` directory and `CMP-test-removalrate.csv` are intentionally **not** mirrored: the SARC paper uses a chronological split inside the training set (early lots → train/val, late lots → held-out test), so the official PHM test split is not consumed by the reproduction pipeline. The PHM test labels were also released only via the separate `PHM16TestValidationAnswers.zip`, which is currently 404 on the PHM site.

## Schema

**`CMP-training-removalrate.csv`** — wafer-level labels:

| Column | Description |
|---|---|
| `WAFER_ID` | Wafer identifier (matches per-wafer trace files) |
| `STAGE` | Polishing stage (`A` or `B`) |
| `AVG_REMOVAL_RATE` | Measured removal rate (target, µm/min) |

**`CMP-data/training/CMP-training-NNN.csv`** — per-wafer process trace (one row per timestamp):

`MACHINE_ID, MACHINE_DATA, TIMESTAMP, WAFER_ID, STAGE, CHAMBER, USAGE_OF_BACKING_FILM, USAGE_OF_DRESSER, USAGE_OF_POLISHING_TABLE, USAGE_OF_DRESSER_TABLE, PRESSURIZED_CHAMBER_PRESSURE, MAIN_OUTER_AIR_BAG_PRESSURE, CENTER_AIR_BAG_PRESSURE, RETAINER_RING_PRESSURE, RIPPLE_AIR_BAG_PRESSURE, USAGE_OF_MEMBRANE, USAGE_OF_PRESSURIZED_SHEET, SLURRY_FLOW_LINE_A, SLURRY_FLOW_LINE_B, SLURRY_FLOW_LINE_C, WAFER_ROTATION, STAGE_ROTATION, HEAD_ROTATION, DRESSING_WATER_STATUS, EDGE_AIR_BAG_PRESSURE`

See the SARC paper (Table II) for the feature/state/action mapping used by the R2R MDP.

## Quickstart (preprocessing)

From the repository root:

```bash
python -m src.data.preprocess_cmp1_r2r
```

This produces `Dataset/CMP1/processed/` with deterministic chronological train/val/test splits. End-to-end paper reproduction:

```bash
bash scripts/reproduce_cmp1.sh
```

## Citation

```bibtex
@misc{phm2016,
  author       = {{PHM Society}},
  title        = {{PHM} 2016 Data Challenge---{CMP} Dataset},
  year         = {2016},
  howpublished = {\url{https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/}},
  note         = {Reproducibility mirror: \url{https://github.com/jk-choi-yonsei/Offline_RL_for_R2R_Control}}
}
```
