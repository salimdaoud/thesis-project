# Thesis Project — Lean Version

**Goal**: Predict sparse linear solver performance from matrix structural/numerical features,
without running the solver.

## Project structure

```
thesis-clean/
├── data/
│   └── processed/           # all CSV datasets (pre-generated)
├── src/
│   └── features/
│       ├── tier1.py         # cheap structural features
│       └── tier2.py         # costly numerical features
├── scripts/
│   └── run_petsc_bench.py   # benchmark runner (requires PETSc)
└── notebooks/
    ├── 01_data_and_eda.ipynb          # feature validation, solver labels, EDA
    ├── 02_convergence_prediction.ipynb # will this solver converge? (classification)
    ├── 03_fast_slow_prediction.ipynb   # which solver is fastest? + Boruta selection
    └── 04_boruta_validation.ipynb      # validate Boruta on synthetic/external data
```

## Notebook pipeline

| # | Notebook | Reads | Writes |
|---|----------|-------|--------|
| 01 | EDA + data construction | `tier1_features.csv`, `tier12_features.csv`, `solver_results.csv` | `ml_dataset_full.csv`, `ml_dataset_converged.csv` |
| 02 | Convergence classification | `ml_dataset_full.csv` | — |
| 03 | Fast/slow classification + Boruta | `ml_dataset_converged.csv` | — |
| 04 | Boruta validation | `ml_dataset_converged.csv`, external data | — |

## Key design decisions

- **Grouped train/test split** (`GroupShuffleSplit` on `matrix_id`) prevents a matrix
  appearing in both train and test sets across solver configs.
- **Consistent log-transforms** applied to `LOG_FEATURES` identically in every notebook
  via a shared constant — this was the root cause of the performance drop when retraining
  with Boruta features in the original code.
- **Fast label**: within 10% of best runtime per matrix (converged runs only).
- **Boruta retraining**: uses the *same* imputed matrix and the *same* train/test indices
  as the full-feature model, ensuring a fair comparison.

## Setup

```bash
conda env create -f thesis-env.yml
conda activate thesis
```
