# scripts/collect_features_mtx.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.io import mmread
import scipy.sparse as sp

from src.features.tier1 import extract_tier1_features


def load_matrix(path: Path) -> sp.spmatrix:
    A = mmread(path)
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    return A.tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="Folder containing .mtx files")
    parser.add_argument("--out_csv", type=str, default="data/processed/tier1_features.csv")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(in_dir.glob("*.mtx")):
        print(f"Processing {p.name}...", flush=True)
        A = load_matrix(p)
        feats = extract_tier1_features(A)
        feats["matrix_id"] = p.stem
        rows.append(feats)
        print(f"✓ {p.name}: nnz={feats['nnz']}")

    df = pd.DataFrame(rows).set_index("matrix_id")
    df.to_csv(out_csv)
    print(f"\nSaved: {out_csv}  (rows={len(df)}, cols={df.shape[1]})")


if __name__ == "__main__":
    main()
