#!/usr/bin/env python
"""Step 03: compare MC grid-marginal samples to released grid-averaged rates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from popsampler.grid_rates import load_rate_grid


DEFAULT_NAMES = ["mass_1", "mass_ratio", "redshift", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]


def normalized_pdf(x: np.ndarray, rate: np.ndarray) -> np.ndarray:
    rate = np.clip(np.asarray(rate, dtype=float), 0.0, np.inf)
    norm = np.trapz(rate, x)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("rate curve has non-positive integral")
    return rate / norm


def weighted_quantile_from_pdf(x: np.ndarray, pdf: np.ndarray, q: float) -> float:
    dx = np.diff(x)
    # trapezoidal cumulative integral
    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dx)
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    parser.add_argument("--outdir", default="validation_outputs", help="Directory containing step-02 outputs")
    parser.add_argument("--bins", type=int, default=150, help="Histogram bins for MC-vs-grid comparison")
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    samples_path = outdir / "grid_marginal_samples.parquet"
    rows_path = outdir / "grid_rows.npy"
    if not samples_path.exists():
        raise FileNotFoundError(f"Run 02_sample_grid_marginals.py first: missing {samples_path}")
    if not rows_path.exists():
        raise FileNotFoundError(f"Run 02_sample_grid_marginals.py first: missing {rows_path}")

    df = pd.read_parquet(samples_path)
    rows = np.load(rows_path)

    diagnostics = []
    for name in DEFAULT_NAMES:
        grid = load_rate_grid(h5, name)
        x = grid.positions
        mean_rate = grid.rates[rows].mean(axis=0)
        pdf = normalized_pdf(x, mean_rate)

        hist, edges = np.histogram(df[name], bins=args.bins, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        target = np.interp(centers, x, pdf)
        l1 = float(np.trapz(np.abs(hist - target), centers))

        mc_mean = float(df[name].mean())
        grid_mean = float(np.trapz(x * pdf, x))
        mc_q05, mc_q50, mc_q95 = [float(df[name].quantile(q)) for q in [0.05, 0.50, 0.95]]
        grid_q05, grid_q50, grid_q95 = [weighted_quantile_from_pdf(x, pdf, q) for q in [0.05, 0.50, 0.95]]

        diagnostics.append(
            {
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
        )

    diag = pd.DataFrame(diagnostics)
    diag_path = outdir / "grid_marginal_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    print(f"wrote {diag_path}")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
