#!/usr/bin/env python
"""Validate analytic GWTC-4 1D marginals against released rate grids.

This is the sharpest formula/convention check in the validation suite. Unlike
sample-vs-grid checks, it has no Monte Carlo noise: for each selected
hyperposterior row it evaluates the analytic model implied by that row, averages
those row-normalized PDFs, and compares the result to the released
``posterior/rates_on_grids`` curves.

Redshift convention:
    The released redshift grid is interpreted as the comoving source-frame
    rate-density shape R(z), not the detector-frame event-redshift PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from popsampler.analytic_marginals import (
    DEFAULT_1D_PARAMETERS,
    analytic_1d_pdf_for_rows,
    compare_curves,
    row_normalized_grid_target,
)
from popsampler.grid_rates import load_rate_grid
from popsampler.models import BBHDefaultModelConfig, GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift
from popsampler.popsummary_io import get_hyperparameter_samples


def load_or_choose_rows(path: Path, hyper_len: int, *, n_hyperrows: int, seed: int) -> np.ndarray:
    if path.exists():
        rows = np.load(path).astype(int)
        if len(rows) > n_hyperrows:
            return rows[:n_hyperrows]
        return rows
    rng = np.random.default_rng(seed)
    return rng.choice(hyper_len, size=n_hyperrows, replace=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    parser.add_argument("--outdir", default="validation_outputs")
    parser.add_argument("--rows", default="validation_outputs/joint_sampler_rows.npy")
    parser.add_argument("--n-hyperrows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--parameters", nargs="+", default=list(DEFAULT_1D_PARAMETERS))
    parser.add_argument("--m1-max", type=float, default=300.0)
    parser.add_argument("--z-max", type=float, default=1.9)
    parser.add_argument(
        "--fail-l1-above",
        type=float,
        default=None,
        help="Optional failure threshold on the L1 curve distance for any parameter.",
    )
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hyper = get_hyperparameter_samples(h5)
    if args.n_hyperrows > len(hyper):
        raise ValueError(f"Requested {args.n_hyperrows} rows, only {len(hyper)} available")
    rows = load_or_choose_rows(Path(args.rows).expanduser(), len(hyper), n_hyperrows=args.n_hyperrows, seed=args.seed)
    np.save(outdir / "analytic_1d_rows.npy", rows)

    model = GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift(
        config=BBHDefaultModelConfig(m1_max=args.m1_max, z_max=args.z_max, warn_if_unvalidated=False)
    )

    diagnostics = []
    curves_dir = outdir / "analytic_1d_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    for name in args.parameters:
        grid = load_rate_grid(h5, name)
        x = grid.positions.astype(float)
        target = row_normalized_grid_target(x, grid.rates[rows])
        model_pdf = analytic_1d_pdf_for_rows(model, hyper, rows, name, x)
        diagnostics.append(compare_curves(name, x, model_pdf, target).as_dict())
        pd.DataFrame({"x": x, "analytic_pdf": model_pdf, "released_grid_pdf": target}).to_csv(
            curves_dir / f"{name}_analytic_vs_grid_curve.csv",
            index=False,
        )

    diag = pd.DataFrame(diagnostics)
    diag_path = outdir / "analytic_1d_marginal_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    print(f"wrote {diag_path}")
    print(diag.to_string(index=False))

    if args.fail_l1_above is not None:
        bad = diag[diag["l1"] > args.fail_l1_above]
        if len(bad):
            raise SystemExit(
                "Analytic/grid L1 threshold failed:\n" + bad[["name", "l1", "mean_abs_diff"]].to_string(index=False)
            )


if __name__ == "__main__":
    main()
