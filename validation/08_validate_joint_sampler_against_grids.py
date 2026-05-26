#!/usr/bin/env python
"""Step 08: compare the named joint sampler's 1D projections to released grids.

This is the first validation gate for the covariance-preserving sampler. The
sampler draws events from named hyperposterior rows, projects those events to the
released one-dimensional parameters, and compares the Monte Carlo histograms to
the corresponding row-averaged `rates_on_grids` products.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from popsampler.grid_rates import load_rate_grid
from popsampler.models import (
    BBHDefaultModelConfig,
    GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift,
)
from popsampler.popsummary_io import get_hyperparameter_samples


PARAMETERS = ["mass_1", "mass_ratio", "redshift", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
SAMPLE_COLUMNS = {
    "mass_1": "mass_1_source",
    "mass_ratio": "mass_ratio",
    "redshift": "redshift",
    "a_1": "a_1",
    "a_2": "a_2",
    "cos_tilt_1": "cos_tilt_1",
    "cos_tilt_2": "cos_tilt_2",
}


def normalized_pdf(x: np.ndarray, rate: np.ndarray) -> np.ndarray:
    rate = np.clip(np.asarray(rate, dtype=float), 0.0, np.inf)
    norm = np.trapz(rate, x)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("rate curve has non-positive integral")
    return rate / norm


def weighted_quantile_from_pdf(x: np.ndarray, pdf: np.ndarray, q: float) -> float:
    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, x))


def compare_parameter(df: pd.DataFrame, h5: Path, rows: np.ndarray, name: str, bins: int) -> dict[str, float | str]:
    grid = load_rate_grid(h5, name)
    x = grid.positions
    pdf = normalized_pdf(x, grid.rates[rows].mean(axis=0))
    values = df[SAMPLE_COLUMNS[name]].to_numpy(dtype=float)

    hist, edges = np.histogram(values, bins=bins, range=(float(x.min()), float(x.max())), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    target = np.interp(centers, x, pdf)
    l1 = float(np.trapz(np.abs(hist - target), centers))

    mc_mean = float(np.mean(values))
    grid_mean = float(np.trapz(x * pdf, x))
    mc_q05, mc_q50, mc_q95 = [float(np.quantile(values, q)) for q in [0.05, 0.50, 0.95]]
    grid_q05, grid_q50, grid_q95 = [weighted_quantile_from_pdf(x, pdf, q) for q in [0.05, 0.50, 0.95]]

    return {
        "name": name,
        "hist_l1": l1,
        "mc_mean": mc_mean,
        "grid_mean": grid_mean,
        "mean_abs_diff": abs(mc_mean - grid_mean),
        "mc_q05": mc_q05,
        "grid_q05": grid_q05,
        "mc_q50": mc_q50,
        "grid_q50": grid_q50,
        "mc_q95": mc_q95,
        "grid_q95": grid_q95,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    parser.add_argument("--outdir", default="validation_outputs", help="Directory for outputs")
    parser.add_argument("--n-hyperrows", type=int, default=200)
    parser.add_argument("--events-per-row", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bins", type=int, default=150)
    parser.add_argument("--z-max", type=float, default=1.9, help="Use 1.9 for released redshift-grid validation; use larger for CE generation")
    parser.add_argument("--no-extrinsics", action="store_true")
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    hyper = get_hyperparameter_samples(h5)
    if args.n_hyperrows > len(hyper):
        raise ValueError(f"Requested {args.n_hyperrows} rows, only {len(hyper)} available")
    rows = rng.choice(len(hyper), size=args.n_hyperrows, replace=False)
    np.save(outdir / "joint_sampler_rows.npy", rows)

    config = BBHDefaultModelConfig(z_max=args.z_max, warn_if_unvalidated=False)
    model = GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift(config=config)

    pieces = []
    for row_idx in tqdm(rows, desc="Sampling named joint model"):
        row = hyper.iloc[int(row_idx)].to_dict()
        block = model.sample(
            row,
            args.events_per_row,
            rng=rng,
            include_extrinsics=not args.no_extrinsics,
        )
        block["hyper_sample_id"] = int(row_idx)
        pieces.append(block)

    samples = pd.concat(pieces, ignore_index=True)
    samples_path = outdir / "joint_sampler_validation_samples.parquet"
    samples.to_parquet(samples_path, index=False)

    diagnostics = [compare_parameter(samples, h5, rows, name, args.bins) for name in PARAMETERS]
    diag = pd.DataFrame(diagnostics)
    diag_path = outdir / "joint_sampler_grid_diagnostics.csv"
    diag.to_csv(diag_path, index=False)

    print(f"wrote {samples_path}")
    print(f"wrote {diag_path}")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
