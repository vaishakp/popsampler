#!/usr/bin/env python
"""Step 01: inspect the GWTC-4 popsummary release file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from popsampler.grid_rates import list_rate_grid_names, load_rate_grid
from popsampler.popsummary_io import get_hyperparameter_samples, list_hdf5_tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    parser.add_argument("--outdir", default="validation_outputs", help="Directory for validation outputs")
    parser.add_argument("--max-tree", type=int, default=500, help="Maximum HDF5 tree entries to write")
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = list_hdf5_tree(h5, max_datasets=args.max_tree)
    tree_path = outdir / "hdf5_tree.txt"
    with tree_path.open("w") as fp:
        for row in rows:
            fp.write(f"{row['type']:7s} {row['name']} shape={row['shape']} dtype={row['dtype']}\n")

    grid_rows = []
    for name in list_rate_grid_names(h5):
        grid = load_rate_grid(h5, name)
        grid_rows.append(
            {
                "name": name,
                "n_hyperposterior_samples": grid.n_hyperposterior_samples,
                "n_grid": len(grid.positions),
                "x_min": float(grid.positions.min()),
                "x_max": float(grid.positions.max()),
                "rate_min": float(grid.rates.min()),
                "rate_max": float(grid.rates.max()),
            }
        )
    grids_df = pd.DataFrame(grid_rows)
    grids_path = outdir / "rate_grids.csv"
    grids_df.to_csv(grids_path, index=False)

    hyper = get_hyperparameter_samples(h5)
    columns_path = outdir / "hyperparameter_columns.txt"
    with columns_path.open("w") as fp:
        fp.write(f"shape: {hyper.shape}\n")
        fp.write(f"column_type: {type(hyper.columns[0]).__name__ if len(hyper.columns) else 'none'}\n")
        for col in hyper.columns:
            fp.write(f"{col}\n")

    print(f"wrote {tree_path}")
    print(f"wrote {grids_path}")
    print(f"wrote {columns_path}")
    print("\n# Rate grids")
    print(grids_df.to_string(index=False))
    print("\n# Hyperparameter table")
    print(f"shape: {hyper.shape}")
    print(f"first columns: {list(hyper.columns[:10])}")


if __name__ == "__main__":
    main()
