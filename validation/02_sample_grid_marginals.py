#!/usr/bin/env python
"""Step 02: sample released 1D marginal rate grids for validation only.

This script intentionally samples the released one-dimensional marginal rate
grids independently. It is useful for checking grid normalization, metadata
alignment, and MC-vs-grid diagnostics. It is not a BBH event sampler and must not
be used for CE catalog generation because it does not preserve joint covariance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from popsampler.grid_rates import load_rate_grid


DEFAULT_NAMES = ["mass_1", "mass_ratio", "redshift", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    parser.add_argument("--outdir", default="validation_outputs", help="Directory for validation outputs")
    parser.add_argument("--n-hyperrows", type=int, default=1000, help="Number of hyperposterior rows to sample")
    parser.add_argument("--events-per-row", type=int, default=1000, help="Number of marginal draws per selected row")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--replace", action="store_true", help="Sample hyperposterior rows with replacement")
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    grids = {name: load_rate_grid(h5, name) for name in DEFAULT_NAMES}
    n_available = grids["mass_1"].n_hyperposterior_samples
    if args.n_hyperrows > n_available and not args.replace:
        raise ValueError(
            f"Requested {args.n_hyperrows} rows without replacement, but only {n_available} are available"
        )

    rows = rng.choice(n_available, size=args.n_hyperrows, replace=args.replace)
    np.save(outdir / "grid_rows.npy", rows)

    pieces = []
    for row in tqdm(rows, desc="Sampling 1D marginal grids"):
        block = {
            name: grid.sample_for_row(
                int(row),
                args.events_per_row,
                rng=rng,
                allow_validation_sampling=True,
            )
            for name, grid in grids.items()
        }
        block["hyper_sample_id"] = np.full(args.events_per_row, int(row), dtype=int)
        block["mass_2"] = block["mass_1"] * block["mass_ratio"]
        block["chi_eff"] = (
            block["a_1"] * block["cos_tilt_1"]
            + block["mass_ratio"] * block["a_2"] * block["cos_tilt_2"]
        ) / (1.0 + block["mass_ratio"])
        pieces.append(pd.DataFrame(block))

    df = pd.concat(pieces, ignore_index=True)
    df["sampler_mode"] = "grid_marginal_validation_only"
    df["preserves_joint_covariance"] = False

    samples_path = outdir / "grid_marginal_samples.parquet"
    summary_path = outdir / "grid_marginal_summary.csv"
    df.to_parquet(samples_path, index=False)
    df.describe().to_csv(summary_path)

    print("WARNING: This output samples 1D marginals independently and is validation-only.")
    print("It must not be used as a covariance-preserving BBH event catalog.")
    print(f"selected {len(rows)} rows from {n_available} available hyperposterior rows")
    print(f"marginal draws per row: {args.events_per_row}")
    print(f"total rows in validation table: {len(df)}")
    print(f"wrote {samples_path}")
    print(f"wrote {summary_path}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
