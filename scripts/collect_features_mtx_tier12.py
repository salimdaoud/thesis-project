# scripts/collect_features_mtx.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from scipy.io import mmread
import scipy.sparse as sp

from src.features.tier1 import extract_tier1_features, Tier1Config
from src.features.tier2 import extract_tier2_features, Tier2Config


def load_matrix(path: Path) -> sp.spmatrix:
    A = mmread(path)
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    return A.tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="Folder containing .mtx files")
    parser.add_argument("--out_csv", type=str, default="data/processed/tier12_features.csv")
    parser.add_argument("--symmetry_sample", type=int, default=200_000, help="Tier1 symmetry sample size")
    parser.add_argument("--power_iters", type=int, default=15, help="Tier2 power iteration steps")
    parser.add_argument("--max_nnz_for_power", type=int, default=5_000_000, help="Skip power iter if nnz exceeds this")
    parser.add_argument("--tiny_diag_tau", type=float, default=1e-12, help="Threshold for tiny diagonal fraction")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tier1_cfg = Tier1Config(symmetry_sample=args.symmetry_sample)
    tier2_cfg = Tier2Config(
        power_iters=args.power_iters,
        max_nnz_for_power=args.max_nnz_for_power,
        tiny_diag_tau=args.tiny_diag_tau,
    )

    rows = []

    # --- Global timer ---
    t_global_start = time.perf_counter()
    times = []


    for p in sorted(in_dir.glob("*.mtx")):
        print(f"Processing {p.name}...", flush=True)
        A = load_matrix(p)

        # --- Tier 1 timing ---
        t1_start = time.perf_counter()
        feats1 = extract_tier1_features(A, cfg=tier1_cfg)
        t1_time = time.perf_counter() - t1_start

        # --- Tier 2 timing ---
        t2_start = time.perf_counter()
        feats2 = extract_tier2_features(A, cfg=tier2_cfg)
        t2_time = time.perf_counter() - t2_start

        total_feat_time = t1_time + t2_time
        times.append(total_feat_time)
        feats = {**feats1, **feats2}
        feats["matrix_id"] = p.stem
        rows.append(feats)


        print(
            f"✓ {p.name}: nnz={feats1['nnz']} "
            f"tier1={t1_time:.3f}s tier2={t2_time:.3f}s total={total_feat_time:.3f}s "
            f"spec_radius_est={feats2.get('spec_radius_est')} "
            f"pseudo_condition_number={feats2.get('pseudo_kappa')}"
        )

    # --- Global timing end ---
    t_global = time.perf_counter() - t_global_start

    df = pd.DataFrame(rows).set_index("matrix_id")
    df.to_csv(out_csv)

    print(f"\nSaved: {out_csv}  (rows={len(df)}, cols={df.shape[1]})")
    print(f"Total feature extraction time: {t_global:.2f} s")
    if times:
        print(f"Average time per matrix: {sum(times)/len(times):.3f} s")



if __name__ == "__main__":
    main()

