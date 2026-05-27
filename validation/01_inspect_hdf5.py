#!/usr/bin/env python
"""Step 01: inspect the popsummary HDF5 tree and hyperposterior sample table."""

from __future__ import annotations

import argparse

import h5py

from common import add_common_args, require_file, resolve_paths, save_json
from popsampler.grid_rates import list_rate_grid_names, load_rate_grid
from popsampler.popsummary_io import get_hyperparameter_samples, list_hdf5_tree


def collect_attrs(h5_path):
    attrs = {}

    def visit(name, obj):
        if obj.attrs:
            attrs[name or "/"] = {k: repr(v) for k, v in obj.attrs.items()}

    with h5py.File(h5_path, "r") as h5:
        if h5.attrs:
            attrs["/"] = {k: repr(v) for k, v in h5.attrs.items()}
        h5.visititems(visit)
    return attrs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--max-tree", type=int, default=80)
    parser.add_argument("--max-columns", type=int, default=40)
    args = parser.parse_args()

    h5, output_dir = resolve_paths(args)
    require_file(h5)

    tree_rows = list_hdf5_tree(h5, max_datasets=args.max_tree)
    tree_txt = output_dir / "01_hdf5_tree.txt"
    with tree_txt.open("w") as fp:
        for row in tree_rows:
            line = f"{row['type']:7s} {row['name']} shape={row['shape']} dtype={row['dtype']}"
            print(line)
            fp.write(line + "\n")

    print("\n# Hyperparameter samples")
    hyper = get_hyperparameter_samples(h5)
    print(f"shape: {hyper.shape}")
    columns = [str(c) for c in hyper.columns]
    print("first columns:")
    for col in columns[: args.max_columns]:
        print(f"  {col}")
    if len(columns) > args.max_columns:
        print(f"  ... {len(columns) - args.max_columns} more")

    grid_names = list_rate_grid_names(h5)
    print("\n# Rate grids")
    grid_summary = {}
    for name in grid_names:
        grid = load_rate_grid(h5, name)
        grid_summary[name] = {
            "positions_shape": list(grid.positions.shape),
            "rates_shape": list(grid.rates.shape),
            "position_min": float(grid.positions.min()),
            "position_max": float(grid.positions.max()),
        }
        print(
            f"{name:12s} positions={grid.positions.shape} rates={grid.rates.shape} "
            f"range=[{grid.positions.min():.6g}, {grid.positions.max():.6g}]"
        )

    payload = {
        "h5": str(h5),
        "hyperparameter_samples_shape": list(hyper.shape),
        "hyperparameter_columns_first": columns[: args.max_columns],
        "rate_grids": grid_summary,
        "attrs": collect_attrs(h5),
    }
    save_json(output_dir / "01_inspection_summary.json", payload)
    print(f"\nwrote {tree_txt}")
    print(f"wrote {output_dir / '01_inspection_summary.json'}")


if __name__ == "__main__":
    main()
