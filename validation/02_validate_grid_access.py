#!/usr/bin/env python
"""Step 02: validate direct access to released one-dimensional rate grids."""

from __future__ import annotations

import argparse

import numpy as np

from common import add_common_args, normalize_pdf, require_file, resolve_paths, save_json, trapezoid
from popsampler.grid_rates import list_rate_grid_names, load_rate_grid


EXPECTED_GRIDS = [
    "mass_1",
    "mass_ratio",
    "redshift",
    "a_1",
    "a_2",
    "cos_tilt_1",
    "cos_tilt_2",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--row", type=int, default=0, help="Hyperposterior row to test")
    parser.add_argument("--n-test", type=int, default=10, help="Number of samples to draw per grid")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    h5, output_dir = resolve_paths(args)
    require_file(h5)
    rng = np.random.default_rng(args.seed)

    names = list_rate_grid_names(h5)
    missing = sorted(set(EXPECTED_GRIDS) - set(names))
    if missing:
        raise RuntimeError(f"Missing expected grids: {missing}. Available grids: {names}")

    summary = {}
    for name in EXPECTED_GRIDS:
        grid = load_rate_grid(h5, name)
        if not (0 <= args.row < grid.n_hyperposterior_samples):
            raise IndexError(f"row={args.row} outside [0, {grid.n_hyperposterior_samples})")
        pdf = normalize_pdf(grid.rate_for_row(args.row), grid.positions)
        samples = grid.sample_for_row(args.row, args.n_test, rng=rng)
        integral = trapezoid(pdf, grid.positions)
        summary[name] = {
            "positions_shape": list(grid.positions.shape),
            "rates_shape": list(grid.rates.shape),
            "row": args.row,
            "normalized_integral": integral,
            "test_samples": samples.tolist(),
        }
        print(f"\n{name}")
        print(f"  positions: {grid.positions.shape}")
        print(f"  rates    : {grid.rates.shape}")
        print(f"  integral : {integral:.16f}")
        print(f"  samples  : {samples}")

    out = output_dir / "02_grid_access_summary.json"
    save_json(out, summary)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
