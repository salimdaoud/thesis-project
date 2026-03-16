from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from petsc4py import PETSc
from scipy.io import mmread
import scipy.sparse as sp


def load_matrix_mtx_to_petsc(path: Path) -> PETSc.Mat:
    A = mmread(str(path))
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = A.tocsr()
    A.sort_indices()

    nrows, ncols = A.shape
    Ap = PETSc.Mat().createAIJ(size=(nrows, ncols), csr=(A.indptr, A.indices, A.data))
    Ap.assemble()
    return Ap


def build_rhs(A: PETSc.Mat) -> tuple[PETSc.Vec, PETSc.Vec]:
    n, _ = A.getSize()
    x_true = PETSc.Vec().createSeq(n)
    x_true.set(1.0)

    b = PETSc.Vec().createSeq(n)
    A.mult(x_true, b)
    return x_true, b

def run_one(A: PETSc.Mat, ksp_type: str, pc_type: str, rtol: float, max_it: int) -> dict:
    n, _ = A.getSize()

    ksp = PETSc.KSP().create()
    prefix = f"{ksp_type}_{pc_type}_"
    ksp.setOptionsPrefix(prefix)

    ksp.setOperators(A)
    ksp.setType(ksp_type)

    pc = ksp.getPC()
    pc.setType(pc_type)

    if pc_type == "ilu":
        opts = PETSc.Options()
        opts[f"{prefix}pc_factor_levels"] = 1
        opts[f"{prefix}pc_factor_shift_type"] = "nonzero"
        opts[f"{prefix}pc_factor_shift_amount"] = 1e-10
        opts[f"{prefix}pc_factor_nonzeros_along_diagonal"] = 1e-10
        opts[f"{prefix}pc_factor_mat_ordering_type"] = "nd"
        opts[f"{prefix}pc_factor_zeropivot"] = 1e-14

    ksp.setTolerances(rtol=rtol, max_it=max_it)
    ksp.setFromOptions()

    x_true, b = build_rhs(A)
    x = PETSc.Vec().createSeq(n)
    x.set(0.0)

    t0 = time.perf_counter()
    ksp.solve(b, x)
    t1 = time.perf_counter()

    its = ksp.getIterationNumber()
    reason = int(ksp.getConvergedReason())
    converged = 1 if reason > 0 else 0

    err = x.copy()
    err.axpy(-1.0, x_true)
    rel_err = err.norm() / (x_true.norm() + 1e-30)

    res_norm = ksp.getResidualNorm()

    return {
        "ksp": ksp_type,
        "pc": pc_type,
        "rtol": rtol,
        "max_it": max_it,
        "converged": converged,
        "reason": reason,
        "iterations": its,
        "solve_time_sec": float(t1 - t0),
        "final_residual_norm": float(res_norm),
        "rel_error_to_ones": float(rel_err),
        "note": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="data/processed/solver_results.csv")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--max_it", type=int, default=500)
    args = parser.parse_args()

    # Suppress PETSc internal stack traces
    PETSc.Sys.pushErrorHandler("ignore")

    in_dir = Path(args.in_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    solver_configs = [
        ("cg", "jacobi"),
        ("gmres", "jacobi"),
        ("bcgs", "jacobi"),
        ("gmres", "ilu"),
        ("bcgs", "ilu"),
        ('minres', 'jacobi'),      
        ('fgmres', 'jacobi'),       
        #('gmres',  'boomeramg'),    
        ('tfqmr',  'jacobi'),       
    ]

    rows: list[dict] = []
    mtx_files = sorted(in_dir.glob("*.mtx"))
    if not mtx_files:
        raise FileNotFoundError(f"No .mtx files found in {in_dir}")

    # --- Global timer: total wall-clock time for the whole script run ---
    t_global_start = time.perf_counter()

    # If you want to compute totals without storing them in CSV:
    per_solve_times: list[float] = []
    per_matrix_totals: list[float] = []

    for p in mtx_files:
        print(f"\n=== {p.name} ===")

        try:
            A = load_matrix_mtx_to_petsc(p)
            nrows, ncols = A.getSize()
            nnz = A.getInfo()["nz_used"]
            print(f"Loaded: shape=({nrows},{ncols}), nnz_used≈{int(nnz)}")
        except Exception as e:
            print(f"!! Failed to load {p.name}: {e}")
            continue

        # --- Per-matrix timer: time to run ALL solver configs for this matrix ---
        matrix_start = time.perf_counter()
        matrix_solve_sum = 0.0

        for ksp_type, pc_type in solver_configs:
            try:
                res = run_one(A, ksp_type, pc_type, rtol=args.rtol, max_it=args.max_it)
            except Exception as e:
                msg = str(e)

                if "error code 63" in msg or "zeropivot" in msg.lower():
                    note = "ILU_FACTORIZATION_FAILED"
                else:
                    note = f"EXCEPTION_{type(e).__name__}"

                res = {
                    "ksp": ksp_type,
                    "pc": pc_type,
                    "rtol": args.rtol,
                    "max_it": args.max_it,
                    "converged": 0,
                    "reason": -998,
                    "iterations": 0,
                    "solve_time_sec": 0.0,
                    "final_residual_norm": float("nan"),
                    "rel_error_to_ones": float("nan"),
                    "note": note,
                }

            # add metadata
            res["matrix_id"] = p.stem
            res["nrows"] = nrows
            res["ncols"] = ncols
            res["nnz_used"] = int(nnz)

            rows.append(res)

            # accumulate times for reporting (this includes 0.0 for skipped/exception)
            per_solve_times.append(float(res["solve_time_sec"]))
            matrix_solve_sum += float(res["solve_time_sec"])

            ok = "✓" if res["converged"] == 1 else "✗"
            note = f"  ({res['note']})" if res.get("note") else ""
            print(
                f"{ok} {ksp_type:5s} + {pc_type:6s}  its={res['iterations']:4d}  "
                f"time={res['solve_time_sec']:.4f}s  reason={res['reason']}{note}"
            )

        matrix_total_wall = time.perf_counter() - matrix_start
        per_matrix_totals.append(matrix_total_wall)

        # Two totals are useful:
        # - matrix_solve_sum: sum of measured solve() calls only
        # - matrix_total_wall: wall-clock including setup overhead inside the loops
        print(
            f"Total (sum solve_time_sec) for {p.name}: {matrix_solve_sum:.4f}s | "
            f"Total wall time for {p.name}: {matrix_total_wall:.4f}s"
        )

    # --- Global timing end ---
    t_global_total = time.perf_counter() - t_global_start

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["matrix_id", "ksp", "pc"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}  (rows={len(df)}, cols={df.shape[1]})")

    # --- Summary timings for later comparison with feature extraction ---
    print(f"Total solver script wall time (all matrices & configs): {t_global_total:.2f} s")

    if per_solve_times:
        avg_solve = sum(per_solve_times) / len(per_solve_times)
        print(f"Average time per solve call: {avg_solve:.4f} s")

    if per_matrix_totals:
        avg_matrix = sum(per_matrix_totals) / len(per_matrix_totals)
        print(f"Average wall time per matrix (all configs): {avg_matrix:.4f} s")


if __name__ == "__main__":
    main()

