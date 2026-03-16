# src/features/tier2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from scipy.sparse.linalg import svds

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Tier2Config:
    # Numerical symmetry: use Frobenius norm ratio ||A - A^T||_F / (||A||_F + eps)
    eps: float = 1e-15

    # Threshold for "tiny diagonal" fraction
    tiny_diag_tau: float = 1e-12

    # Power iteration settings for spectral radius estimate
    power_iters: int = 15
    power_seed: int = 0

    # If matrix is huge, skip expensive-ish computations
    max_nnz_for_power: int = 5_000_000  # adjust if needed


def _safe_float(x) -> float:
    """
    Safely convert scalar to float.
    - For complex values, discard tiny imaginary parts explicitly.
    - For non-finite or invalid values, return NaN.
    """
    try:
        if np.iscomplexobj(x):
            return float(np.real(x))
        return float(x)
    except Exception:
        return float("nan")



def row_norm2_stats(A: sp.spmatrix) -> Dict[str, float]:
    """Row 2-norm statistics: mean/std/min/max of ||A[i,:]||_2."""
    A = A.tocsr()
    # Row 2-norms: sqrt(sum_j a_ij^2)
    row_sq = np.asarray(A.multiply(A).sum(axis=1)).ravel()
    row_norm = np.sqrt(row_sq)

    if row_norm.size == 0:
        return {
            "row_norm2_mean": float("nan"),
            "row_norm2_std": float("nan"),
            "row_norm2_min": float("nan"),
            "row_norm2_max": float("nan"),
        }

    return {
        "row_norm2_mean": _safe_float(row_norm.mean()),
        "row_norm2_std": _safe_float(row_norm.std(ddof=0)),
        "row_norm2_min": _safe_float(row_norm.min()),
        "row_norm2_max": _safe_float(row_norm.max()),
    }


def col_norm2_stats(A: sp.spmatrix) -> Dict[str, float]:
    """Column 2-norm statistics: mean/std/min/max of ||A[:,j]||_2."""
    A = A.tocsc()
    col_sq = np.asarray(A.multiply(A).sum(axis=0)).ravel()
    col_norm = np.sqrt(col_sq)

    if col_norm.size == 0:
        return {
            "col_norm2_mean": float("nan"),
            "col_norm2_std": float("nan"),
            "col_norm2_min": float("nan"),
            "col_norm2_max": float("nan"),
        }

    return {
        "col_norm2_mean": _safe_float(col_norm.mean()),
        "col_norm2_std": _safe_float(col_norm.std(ddof=0)),
        "col_norm2_min": _safe_float(col_norm.min()),
        "col_norm2_max": _safe_float(col_norm.max()),
    }


def diag_health_features(A: sp.spmatrix, cfg: Tier2Config) -> Dict[str, float]:
    """Features related to diagonal magnitude health."""
    d = A.diagonal()
    if d.size == 0:
        return {
            "diag_abs_min": float("nan"),
            "diag_abs_max": float("nan"),
            "tiny_diag_fraction": float("nan"),
        }

    diag_abs = np.abs(d)
    return {
        "diag_abs_min": _safe_float(diag_abs.min()),
        "diag_abs_max": _safe_float(diag_abs.max()),
        "tiny_diag_fraction": _safe_float(np.mean(diag_abs < cfg.tiny_diag_tau)),
    }


def numerical_symmetry_fro_ratio(A: sp.spmatrix, cfg: Tier2Config) -> float:
    """
    Numerical symmetry measure:
      s = ||A - A^T||_F / (||A||_F + eps)
    0 => perfectly symmetric values; larger => more non-symmetric.
    """
    # Fro norm of sparse: sqrt(sum of squares of data)
    A = A.tocsr()
    fro_A = np.sqrt(np.sum(np.abs(A.data) ** 2))
    # Compute A - A^T (sparse)
    D = A - A.T
    D = D.tocsr()
    fro_D = np.sqrt(np.sum(np.abs(D.data) ** 2))
    return _safe_float(fro_D / (fro_A + cfg.eps))


def spectral_radius_est_power(A: sp.spmatrix, cfg: Tier2Config) -> float:
    """
    Cheap spectral radius estimate via power iteration:
      rho(A) ~ ||A v|| / ||v|| after a few iterations.
    Returns NaN if skipped due to size.
    """
    if A.nnz > cfg.max_nnz_for_power:
        return float("nan")

    A = A.tocsr()
    n = A.shape[0]
    if n == 0:
        return float("nan")

    rng = np.random.default_rng(cfg.power_seed)
    v = rng.standard_normal(n).astype(float)

    # Avoid zero vector
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        v[0] = 1.0
        v_norm = 1.0

    for _ in range(max(1, cfg.power_iters)):
        w = A @ v
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            return 0.0
        v = w / w_norm

    # Rayleigh-ish scale estimate
    w = A @ v
    rho = np.linalg.norm(w) / (np.linalg.norm(v) + cfg.eps)
    return _safe_float(rho)


def sigma_max_est_power(A: sp.spmatrix, iters: int = 20, seed: int = 0, eps: float = 1e-15) -> float:
    """
    Estimate ||A||_2 (largest singular value) using power iteration on A^T A:
      sigma_max(A) = sqrt(lambda_max(A^T A))

    This is more appropriate than using A v directly when A is non-symmetric.
    """
    A = A.tocsr()
    ncols = A.shape[1]
    if ncols == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(ncols).astype(float)

    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        v[0] = 1.0
        v_norm = 1.0
    v /= v_norm

    # Power iterations on (A^T A)
    for _ in range(max(1, iters)):
        w = A @ v            # shape (nrows,)
        z = A.T @ w          # shape (ncols,)
        z_norm = np.linalg.norm(z)
        if z_norm == 0:
            return 0.0
        v = z / z_norm

    # Rayleigh estimate for lambda_max(A^T A)
    w = A @ v
    z = A.T @ w
    lam = lam = float(np.real(np.vdot(v, z))) # v^T (A^T A) v
    lam = max(lam, 0.0)
    return _safe_float(np.sqrt(lam))


def row_norm2_spread(A: sp.spmatrix, eps: float = 1e-30) -> float:
    """
    Spread of row 2-norms:
      max_i ||A[i,:]||_2 / min_{i:||A[i,:]||_2>0} ||A[i,:]||_2
    Works for real and complex matrices.
    """
    A = A.tocsr()

    # Use |a_ij|^2 for complex safety
    row_sq = np.asarray(np.abs(A).power(2).sum(axis=1)).ravel()
    row_norm = np.sqrt(row_sq)

    if row_norm.size == 0:
        return float("nan")

    nz = row_norm[row_norm > 0]
    if nz.size == 0:
        return float("inf")

    return _safe_float(nz.max() / (nz.min() + eps))


def pseudo_condition_number(A: sp.spmatrix, cfg: Tier2Config) -> Dict[str, float]:
    """
    Pseudo condition number:
      pseudo_kappa = sigma_max_est(A) * row_norm2_spread(A)

    - sigma_max_est via power iteration on A^T A (robust for non-symmetric A)
    - spread captures scaling/heterogeneity
    """
    sig = sigma_max_est_power(A, iters=cfg.power_iters, seed=cfg.power_seed, eps=cfg.eps)
    spread = row_norm2_spread(A)
    pseudo = float("nan")
    if np.isfinite(sig) and np.isfinite(spread):
        pseudo = _safe_float(sig * spread)

    return {
        "sigma_max_est_power": _safe_float(sig),
        "row_norm2_spread": _safe_float(spread),
        "pseudo_kappa": _safe_float(pseudo),
    }


def extract_tier2_features(A: sp.spmatrix, cfg: Optional[Tier2Config] = None) -> Dict[str, Any]:
    """
    Tier-2 numerical/approximate features for a sparse matrix A.
    """
    if cfg is None:
        cfg = Tier2Config()
    if not sp.issparse(A):
        raise TypeError("A must be a SciPy sparse matrix")

    feats: Dict[str, Any] = {}
    feats.update(row_norm2_stats(A))
    feats.update(col_norm2_stats(A))
    feats.update(diag_health_features(A, cfg))
    feats["symmetry_numeric_fro_ratio"] = numerical_symmetry_fro_ratio(A, cfg)
    feats["spec_radius_est"] = spectral_radius_est_power(A, cfg)
    feats.update(pseudo_condition_number(A, cfg))
    return feats
