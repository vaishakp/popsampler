#!/usr/bin/env python
"""Step 03: generate posterior-predictive samples from released 1D rate grids.

This script samples each one-dimensional marginal independently conditional on
one hyperposterior row. It is therefore a validation tool for the released
`rates_on_grids` products, not a full joint/correlated BBH population sampler.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common import add_common_args, require_file, resolve_paths, save_json
from popsampler.grid_rates import load_rate_grid


GRID_NAMES = [
    "mass_1",
    "mass_ratio",
    "redshift",
    "a_1",
    "a_2",
    "cos_tilt_1",
    "cos_tilt_2",
]


def choose_rows(n_available: int, n_hyperrows: int | None, rng: np.random.Generator) -> np.ndarray:
    if n_hyperrows is None or n_hyperrows == n_available:
        return np.arange(n_available, dtype=int)
    if n_hyperrows <= 0:
        raise ValueError("--n-hyperrows must be positive")
    if n_hyperrows > n_available:
        raise ValueError(f"Requested {n_hyperrows} rows but only {n_available} are available")
    return np.sort(rng.choice(n_available, size=n_hyperrows, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n-hyperrows", type=int, default=1000, help="Number of hyperposterior rows to use. Use -1 for all rows.")
    parser.add_argument("--events-per-row", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="grid_marginal_samples.parquet")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    h5, output_dir = resolve_paths(args)
    require_file(h5)
    rng = np.random.default_rng(args.seed)

    grids = {name: load_rate_grid(h5, name) for name in GRID_NAMES}
    n_available = grids["mass_1"].n_hyperposterior_samples
    n_hyperrows = None if args.n_hyperrows == -1 else args.n_hyperrows
    rows = choose_rows(n_available, n_hyperrows, rng)

    if args.events_per_row <= 0:
        raise ValueError("--events-per-row must be positive")

    pieces = []
    iterator = tqdm(rows, desc="Sampling grid marginals", disable=args.no_progress)
    for row in iterator:
        data = {name: grid.sample_for_row(int(row), args.events_per_row, rng=rng) for name, grid in grids.items()}
        df = pd.DataFrame(data)
        df["mass_2"] = df["mass_1"] * df["mass_ratio"]
        df["chi_eff"] = (
            df["a_1"] * df["cos_tilt_1"]
            + df["mass_ratio"] * df["a_2"] * df["cos_tilt_2"]
        ) / (1.0 + df["mass_ratio"])
        df["hyper_sample_id"] = int(row)
        pieces.append(df)

    out_df = pd.concat(pieces, ignore_index=True)
    output = output_dir / args.output
    out_df.to_parquet(output, index=False)
    np.save(output_dir / "grid_marginal_rows.npy", rows)

    summary = {
        "h5": str(h5),
        "output": str(output),
        "n_available_hyperrows": int(n_available),
        "n_used_hyperrows": int(len(rows)),
        "events_per_row": int(args.events_per_row),
        "n_total_samples": int(len(out_df)),
        "seed": int(args.seed),
        "columns": list(out_df.columns),
    }
    save_json(output_dir / "03_grid_marginal_sampling_summary.json", summary)

    print(out_df.describe())
    print(f"\nwrote {output}")
    print(f"wrote {output_dir / 'grid_marginal_rows.npy'}")
    print(f"wrote {output_dir / '03_grid_marginal_sampling_summary.json'}")


if __name__ == "__main__":
    main()
