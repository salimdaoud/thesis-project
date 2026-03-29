# Meta-Feature Engineering and Explainability in Solver Performance Prediction

Bachelor's Thesis in Informatics — Technische Universität München  
**Author:** Salim Daoud  
**Supervisor:** Univ.-Prof. Dr. Hans-Joachim Bungartz  
**Advisor:** M.Sc. Hayden Liu Weng  

---

## Overview

This project investigates whether inexpensive matrix meta-features can be used to predict 
the performance of iterative Krylov solvers for sparse linear systems using machine learning.
A two-tier feature extraction pipeline is developed and evaluated on 808 sparse matrices 
from the SuiteSparse Matrix Collection. The trained models achieve a speedup of 13.8× 
over empirical trial-and-error solver selection.

---

## Project Structure
```
.
├── data/
│   └── processed/             # Extracted features and solver benchmark results
│       ├── tier1_features.csv            # Tier 1 structural features
│       ├── tier12_features.csv           # Combined Tier 1 + Tier 2 features
│       ├── solver_results.csv            # Raw PETSc benchmark results
│       ├── ml_dataset_full.csv           # Full ML dataset (all solver runs)
│       ├── ml_dataset_converged.csv      # Filtered dataset (converged runs only)
│       └── Testing_with_other_dataset/   # Validation on external dataset
│           ├── properties_all_new.csv
│           └── SampleData_0.10_threshold.csv
│
├── notebooks/
│   ├── 01_data_and_eda.ipynb             # Exploratory data analysis
│   ├── 02_convergence_prediction.ipynb   # Convergence classification models + Boruta + SHAP
│   ├── 03_fast_slow_prediction.ipynb     # Fast/slow runtime classification models + Boruta + SHAP
│   ├── 04_boruta_validation.ipynb        # validate Boruta on synthetic/external data
│   └── results/
│       └── figures/                      # Generated plots and figures
│
├── scripts/
│   ├── download_diverse_matrices.py      # Download matrices from SuiteSparse
│   ├── collect_features_mtx.py           # Extract Tier 1 features from .mtx files
│   ├── collect_features_mtx_tier12.py    # Extract Tier 1 + Tier 2 features
│   └── run_petsc_bench.py                # Run PETSc solver benchmarks
│
├── src/
│   └── features/
│       ├── tier1.py                      # Tier 1 feature extraction functions
│       └── tier2.py                      # Tier 2 feature extraction functions
│
├── thesis-env.yml                        # Conda environment specification
└── README.md
```

---

## Setup

### 1. Create the Conda environment
```bash
conda env create -f thesis-env.yml
conda activate thesis-env
```

### 2. Download matrices
```bash
python scripts/download_diverse_matrices.py
```

This downloads sparse matrices from the SuiteSparse Matrix Collection into 
`data/matrices_new/`.

---

## Reproducing the Results

### Step 1 — Extract matrix features
```bash
# Tier 1 features only
python scripts/collect_features_mtx.py

# Tier 1 + Tier 2 features
python scripts/collect_features_mtx_tier12.py
```

Output:
- `collect_features_mtx.py` saves to `data/processed/tier1_features.csv`
- `collect_features_mtx_tier12.py` saves to `data/processed/tier12_features.csv`

### Step 2 — Run solver benchmarks
```bash
python scripts/run_petsc_bench.py
```

Requires a working PETSc installation. Results are saved to 
`data/processed/solver_results.csv`.

### Step 3 — Run the notebooks in order

| Notebook | Description |
|---|---|
| `01_data_and_eda.ipynb` | Dataset statistics and exploratory analysis |
| `02_convergence_prediction.ipynb` | Train and evaluate convergence classifiers |
| `03_fast_slow_prediction.ipynb` | Train and evaluate fast/slow classifiers |
| `04_boruta_validation.ipynb` | validate Boruta on synthetic/external data |

---

## Key Results

| Task | Model | Accuracy | F1 (minority class) |
|---|---|---|---|
| Convergence prediction | Random Forest | 0.901 | 0.728 |
| Fast/slow prediction | HistGradientBoosting | 0.829 | 0.740 |

The two-stage prediction pipeline achieves a **13.8× speedup** over empirically 
running all eight solver configurations.

---

## Dependencies

All dependencies are listed in `thesis-env.yml`. Key libraries include:

- `scikit-learn` — machine learning models
- `shap` — SHAP explainability
- `boruta` — Boruta feature selection
- `scipy` — sparse matrix handling
- `petsc4py` — PETSc Python bindings (required for benchmarking)
- `jupyter` — notebooks

---

## Repository

The full implementation is available at:  
[https://github.com/salimdaoud/thesis-project](https://github.com/salimdaoud/thesis-project)