"""
download_diverse_matrices.py
----------------------------
Downloads new matrices from SuiteSparse that are structurally DIFFERENT
from the ones you already have.

Strategy: target specific application domains that are underrepresented
in the current dataset, which is dominated by circuit/FPGA/graph matrices.

Usage:
    python scripts/download_diverse_matrices.py \
        --existing_features data/processed/tier1_features.csv \
        --out_dir data/matrices_new \
        --n_target 150

Requirements:
    pip install ssgetpy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import ssgetpy

# ── Target domains ────────────────────────────────────────────────────────────
# These are SuiteSparse group prefixes that represent domains currently
# underrepresented in the dataset (which is heavy on circuit/FPGA/graph).
# Each entry is (group_keyword, reason).
TARGET_DOMAINS = [
    # Computational fluid dynamics — non-symmetric, large bandwidth
    ("DNVS",          "CFD / structural"),
    ("Boeing",        "structural engineering"),
    ("Nasa",          "NASA structural"),
    ("Simon",         "CFD"),
    ("Bai",           "eigenvalue problems"),
    ("Wissgott",      "CFD brain"),
    ("FEMLAB",        "FEM multiphysics"),
    ("Pothen",        "mesh partitioning"),
    ("Hamm",          "optimization"),
    ("Lourakis",      "computer vision"),
    ("Williams",      "scientific"),
    ("Sandia",        "Sandia labs"),
    ("Oberwolfach",   "model reduction"),
    ("UTEP",          "optimization LP"),
    ("Botonakis",     "CFD"),
    ("GHS_psdef",     "positive definite structural"),
    ("GHS_indef",     "indefinite — tests minres"),
    ("HB",            "Harwell-Boeing classic set"),
    ("Cylshell",      "shell structure FEM"),
    ("Andrews",       "optimization"),
]

# ── Size bounds ───────────────────────────────────────────────────────────────
# Stay in a range your PETSc benchmark can handle in reasonable time.
N_MIN = 500
N_MAX = 50_000
NNZ_MAX = 2_000_000


def load_existing_names(features_csv: Path) -> set[str]:
    """Return the set of matrix names already in your dataset."""
    if not features_csv.exists():
        print(f"Warning: {features_csv} not found — assuming no existing matrices.")
        return set()
    df = pd.read_csv(features_csv, index_col=0)
    return set(df.index.str.lower())


def search_domain(keyword: str, n_min: int, n_max: int) -> list:
    """Search SuiteSparse for matrices from a given group keyword."""
    try:
        results = ssgetpy.search(
            group=keyword,
            rowbounds=(n_min, n_max),
            colbounds=(n_min, n_max),
            limit=500,
        )
        return list(results)
    except Exception as e:
        print(f"  Search failed for '{keyword}': {e}")
        return []


def is_square(m) -> bool:
    try:
        return int(m.rows) == int(m.cols)
    except Exception:
        return False


def nnz_ok(m, nnz_max: int) -> bool:
    try:
        return int(m.nnz) <= nnz_max
    except Exception:
        return True  # unknown — try anyway


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing_features",
                        default="data/processed/tier1_features.csv",
                        help="Path to your current tier1_features.csv")
    parser.add_argument("--out_dir",
                        default="data/matrices_new",
                        help="Where to save new .mtx files")
    parser.add_argument("--n_target", type=int, default=150,
                        help="How many new matrices to download")
    parser.add_argument("--n_min", type=int, default=N_MIN)
    parser.add_argument("--n_max", type=int, default=N_MAX)
    parser.add_argument("--nnz_max", type=int, default=NNZ_MAX)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load existing matrix names to avoid re-downloading ───────────────────
    existing = load_existing_names(Path(args.existing_features))
    print(f"Existing matrices to skip: {len(existing)}")

    # ── Collect candidates across all target domains ──────────────────────────
    print("\nSearching SuiteSparse across target domains...")
    candidates = []
    seen_names = set()

    for keyword, reason in TARGET_DOMAINS:
        results = search_domain(keyword, args.n_min, args.n_max)
        added = 0
        for m in results:
            name_lower = m.name.lower()
            if name_lower in existing:
                continue                     # already have it
            if name_lower in seen_names:
                continue                     # duplicate across domains
            if not is_square(m):
                continue                     # only square matrices
            if not nnz_ok(m, args.nnz_max):
                continue                     # too dense

            candidates.append((m, reason))
            seen_names.add(name_lower)
            added += 1

        print(f"  {keyword:25s} ({reason:30s}) → {added} new candidates")

    print(f"\nTotal unique candidates: {len(candidates)}")

    if len(candidates) == 0:
        print("No candidates found. Try relaxing --n_min / --n_max / --nnz_max.")
        return

    # ── Sort by size for a balanced mix (small → large) ──────────────────────
    def safe_rows(item):
        try:
            return int(item[0].rows)
        except Exception:
            return 0

    candidates.sort(key=safe_rows)

    # Interleave: take every k-th candidate so we get size diversity
    # rather than all small or all large matrices
    step = max(1, len(candidates) // args.n_target)
    selected = candidates[::step][: args.n_target]
    print(f"Selected {len(selected)} matrices (every {step}-th by size)\n")

    # ── Download ──────────────────────────────────────────────────────────────
    ok = fail = skip = 0
    download_log = []

    for i, (m, domain) in enumerate(selected, 1):
        name = f"{m.group}/{m.name}"
        dst_check = out_dir / f"{m.name}.mtx"

        if dst_check.exists():
            print(f"[{i:3d}/{len(selected)}] already exists — {name}")
            skip += 1
            download_log.append({"name": m.name, "group": m.group,
                                  "domain": domain, "status": "skipped",
                                  "rows": m.rows, "nnz": m.nnz})
            continue

        try:
            m.download(format="MM", destpath=str(out_dir), extract=True)
            print(f"[{i:3d}/{len(selected)}] ✓  {name:50s}"
                  f"  rows={m.rows}  nnz={m.nnz}  ({domain})")
            ok += 1
            download_log.append({"name": m.name, "group": m.group,
                                  "domain": domain, "status": "ok",
                                  "rows": m.rows, "nnz": m.nnz})
            time.sleep(0.3)   # be polite to the server

        except Exception as e:
            print(f"[{i:3d}/{len(selected)}] ✗  {name}  → {e}")
            fail += 1
            download_log.append({"name": m.name, "group": m.group,
                                  "domain": domain, "status": f"failed:{e}",
                                  "rows": m.rows, "nnz": m.nnz})

    # ── Summary ───────────────────────────────────────────────────────────────
    log_path = out_dir / "download_log.csv"
    pd.DataFrame(download_log).to_csv(log_path, index=False)

    print(f"\n{'─'*55}")
    print(f"Downloaded:  {ok}")
    print(f"Skipped:     {skip}  (already existed)")
    print(f"Failed:      {fail}")
    print(f"Output dir:  {out_dir}")
    print(f"Log:         {log_path}")
    print(f"\nNext steps:")
    print(f"  1. python scripts/build_benchmark_set.py --sources {out_dir}")
    print(f"  2. python scripts/collect_features_mtx_tier12.py --in_dir {out_dir}")
    print(f"  3. python scripts/run_petsc_bench.py --in_dir data/bench/raw")


if __name__ == "__main__":
    main()
