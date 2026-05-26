"""Inspect a popsummary HDF5 file."""

from __future__ import annotations

import argparse

from popsampler.popsummary_io import get_hyperparameter_samples, list_hdf5_tree


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Popsummary HDF5 file")
    parser.add_argument("--max-tree", type=int, default=50, help="Maximum HDF5 tree entries to print")
    parser.add_argument("--max-columns", type=int, default=200, help="Maximum hyperparameter columns to print")
    args = parser.parse_args(argv)

    print("# HDF5 tree")
    for row in list_hdf5_tree(args.file, max_datasets=args.max_tree):
        print(f"{row['type']:7s} {row['name']} shape={row['shape']} dtype={row['dtype']}")

    print("\n# Hyperparameter samples")
    df = get_hyperparameter_samples(args.file)
    print(f"shape: {df.shape}")
    for col in list(df.columns)[: args.max_columns]:
        print(col)
    if len(df.columns) > args.max_columns:
        print(f"... {len(df.columns) - args.max_columns} more columns")


if __name__ == "__main__":
    main()
