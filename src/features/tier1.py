# src/features/tier1.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class Tier1Config:
    symmetry_sample: int = 200_000  # sample size for fast structural symmetry estimate


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def structural_symmetry_score(A: sp.spmatrix, cfg: Tier1Config) -> float:
    """
    Approximate structural symmetry:
    fraction of sampled nonzeros (i,j) for which (j,i) is also nonzero.

    Returns value in [0, 1], where 1 means structurally symmetric.
    """
    A = A.tocoo()
    nnz = A.nnz
    if nnz == 0:
        return 1.0

    # Build a fast membership structure using CSR for O(1) row access
    Acsr = A.tocsr()

    # Sample indices of nonzeros to avoid O(nnz) checks for huge matrices
    m = min(cfg.symmetry_sample, nnz)
    idx = np.random.default_rng(0).choice(nnz, size=m, replace=False)

    rows = A.row[idx]
    cols = A.col[idx]

    hits = 0
    for i, j in zip(rows, cols):
        # Check if (j, i) exists by scanning row j in CSR
        row_start = Acsr.indptr[j]
        row_end = Acsr.indptr[j + 1]
        row_cols = Acsr.indices[row_start:row_end]
        # binary search if sorted (CSR indices usually sorted, but not guaranteed)
        # We'll do a fast membership check using np.searchsorted
        k = np.searchsorted(row_cols, i)
        if k < row_cols.size and row_cols[k] == i:
            hits += 1

    return hits / m


def diag_dominance_fraction(A: sp.spmatrix) -> float:
    """
    Fraction of rows i such that |a_ii| >= sum_{j != i} |a_ij|.
    """
    A = A.tocsr()
    nrows, ncols = A.shape
    if nrows == 0:
        return float("nan")

    absA = abs(A)
    row_abs_sum = np.asarray(absA.sum(axis=1)).ravel()
    diag = np.zeros(nrows, dtype=float)

    # Extract diagonal (works for rectangular too, but limit to min dimension)
    d = A.diagonal()
    diag[: min(nrows, d.size)] = np.abs(d[: min(nrows, d.size)])

    off_sum = row_abs_sum - diag
    # Condition: diag >= off_sum
    return float(np.mean(diag >= off_sum))


def extract_tier1_features(A: sp.spmatrix, cfg: Tier1Config | None = None) -> Dict[str, Any]:
    """
    Compute Tier-1 baseline features for a sparse matrix A (SciPy sparse).
    Returns a dict of feature_name -> value.
    """
    if cfg is None:
        cfg = Tier1Config()

    if not sp.issparse(A):
        raise TypeError("A must be a SciPy sparse matrix")

    nrows, ncols = A.shape
    nnz = A.nnz
    density = _safe_float(nnz / (nrows * ncols)) if nrows > 0 and ncols > 0 else float("nan")

    # nnz-per-row stats
    A_csr = A.tocsr()
    nnz_per_row = np.diff(A_csr.indptr).astype(float)
    nnz_per_row_mean = _safe_float(nnz_per_row.mean()) if nrows > 0 else float("nan")
    nnz_per_row_std = _safe_float(nnz_per_row.std(ddof=0)) if nrows > 0 else float("nan")
    nnz_per_row_min = _safe_float(nnz_per_row.min()) if nrows > 0 else float("nan")
    nnz_per_row_max = _safe_float(nnz_per_row.max()) if nrows > 0 else float("nan")

    # diagonal stats
    diag = A.diagonal()  # length min(nrows, ncols)
    if diag.size == 0:
        zero_diag_fraction = float("nan")
        diag_abs_mean = float("nan")
        diag_abs_std = float("nan")
    else:
        diag_abs = np.abs(diag)
        zero_diag_fraction = float(np.mean(diag_abs == 0.0))
        diag_abs_mean = _safe_float(diag_abs.mean())
        diag_abs_std = _safe_float(diag_abs.std(ddof=0))

    # norms (sparse)
    # 1-norm and inf-norm are available via spla.norm
    norm_1 = _safe_float(spla.norm(A, ord=1))
    norm_inf = _safe_float(spla.norm(A, ord=np.inf))

    # dominance + symmetry
    diag_dom_frac = diag_dominance_fraction(A)
    symm_struct = structural_symmetry_score(A, cfg)

    return {
        "nrows": int(nrows),
        "ncols": int(ncols),
        "nnz": int(nnz),
        "density": density,
        "nnz_per_row_mean": nnz_per_row_mean,
        "nnz_per_row_std": nnz_per_row_std,
        "nnz_per_row_min": nnz_per_row_min,
        "nnz_per_row_max": nnz_per_row_max,
        "zero_diag_fraction": zero_diag_fraction,
        "diag_abs_mean": diag_abs_mean,
        "diag_abs_std": diag_abs_std,
        "diag_dominance_fraction": diag_dom_frac,
        "norm_1": norm_1,
        "norm_inf": norm_inf,
        "symmetry_score_struct": symm_struct,
    }
