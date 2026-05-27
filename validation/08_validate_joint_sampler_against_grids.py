#!/usr/bin/env python
"""Step 08: validate the named joint sampler against released 1D grids.

This is the main stochastic validation gate for the covariance-preserving
sampler. The sampler draws a fixed number of events from each selected
hyperposterior row, so the default validation target is the equal-row average of
per-row normalized released grids.

Important redshift convention:
    The released ``rates_on_grids/redshift`` product for the GWTC-4 BBH file is
    a comoving source-frame rate-density grid, R(z) ∝ (1 + z)^lambda. It is not
    the detector-frame event-redshift sampling PDF, which includes
    ``dVc/dz / (1 + z)``. Therefore this script compares event samples to the
    released grids for masses and spins only. Redshift is written as a separate
    deterministic rate-density check.
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


SAMPLE_DISTRIBUTION_PARAMETERS = ["mass_1", "mass_ratio", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
SAMPLE_COLUMNS = {
    "mass_1": "mass_1_source",
    "mass_ratio": "mass_ratio",
    "a_1": "a_1",
    "a_2": "a_2",
    "cos_tilt_1": "cos_tilt_1",
    "cos_tilt_2": "cos_tilt_2",
}


def normalized_pdf(x: np.ndarray, rate: np.ndarray) -> np.ndarray:
    rate = np.clip(np.asarray(rate, dtype=float), 0.0, np.inf)
    norm = np.trapezoid(rate, x)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("rate curve has non-positive integral")
    return rate / norm


def target_pdf_from_grid(x: np.ndarray, rates: np.ndarray, *, target_weighting: str) -> np.ndarray:
    """Construct a normalized target from released per-row rate curves.

    equal_row_pdf:
        Normalize each hyperposterior row's rate curve first, then average the
        PDFs. This matches validation runs with a fixed `events_per_row`.

    rate_weighted:
        Average the unnormalized rate curves first, then normalize. This matches
        a Poisson/rate-weighted catalog ensemble.
    """
    rates = np.clip(np.asarray(rates, dtype=float), 0.0, np.inf)
    if target_weighting == "rate_weighted":
        return normalized_pdf(x, rates.mean(axis=0))
    if target_weighting == "equal_row_pdf":
        norms = np.trapezoid(rates, x, axis=1)
        good = np.isfinite(norms) & (norms > 0)
        if not np.any(good):
            raise ValueError("No selected grid rows have positive finite integrals")
        pdfs = rates[good] / norms[good, None]
        return normalized_pdf(x, pdfs.mean(axis=0))
    raise ValueError(f"Unknown target_weighting={target_weighting!r}")


def weighted_quantile_from_pdf(x: np.ndarray, pdf: np.ndarray, q: float) -> float:
    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, x))


def compare_sample_parameter(
    df: pd.DataFrame,
    h5: Path,
    rows: np.ndarray,
    name: str,
    bins: int,
    *,
    target_weighting: str,
) -> dict[str, float | str]:
    """Compare sampled event parameters to released 1D grids.

    Do not use this for redshift: the released redshift grid is a comoving
    rate-density curve, not the detector-frame event-redshift distribution.
    """
    grid = load_rate_grid(h5, name)
    x = grid.positions
    pdf = target_pdf_from_grid(x, grid.rates[rows], target_weighting=target_weighting)
    values = df[SAMPLE_COLUMNS[name]].to_numpy(dtype=float)

    hist, edges = np.histogram(values, bins=bins, range=(float(x.min()), float(x.max())), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    target = np.interp(centers, x, pdf)
    l1 = float(np.trapezoid(np.abs(hist - target), centers))

    mc_mean = float(np.mean(values))
    grid_mean = float(np.trapezoid(x * pdf, x))
    mc_q05, mc_q50, mc_q95 = [float(np.quantile(values, q)) for q in [0.05, 0.50, 0.95]]
    grid_q05, grid_q50, grid_q95 = [weighted_quantile_from_pdf(x, pdf, q) for q in [0.05, 0.50, 0.95]]

    return {
        "name": name,
        "comparison_type": "event_sample_vs_released_grid",
        "target_weighting": target_weighting,
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


def redshift_rate_density_pdf(z: np.ndarray, lamb: float) -> np.ndarray:
    """Normalized comoving source-frame rate-density shape R(z) ∝ (1+z)^lambda."""
    return normalized_pdf(z, np.power(1.0 + np.asarray(z, dtype=float), float(lamb)))


def compare_redshift_rate_density(
    hyper: pd.DataFrame,
    h5: Path,
    rows: np.ndarray,
    *,
    target_weighting: str,
) -> dict[str, float | str]:
    """Compare released redshift grid to paper-defined comoving rate density.

    This deliberately does not use the sampled event redshifts. The event
    redshift distribution includes volume and detector-frame time-dilation
    factors, while the released grid stores the shape of R(z).
    """
    grid = load_rate_grid(h5, "redshift")
    z = grid.positions
    target = target_pdf_from_grid(z, grid.rates[rows], target_weighting=target_weighting)
    model = normalized_pdf(
        z,
        np.mean([redshift_rate_density_pdf(z, float(hyper.iloc[int(idx)]["lamb"])) for idx in rows], axis=0),
    )

    model_mean = float(np.trapezoid(z * model, z))
    grid_mean = float(np.trapezoid(z * target, z))
    model_q05, model_q50, model_q95 = [weighted_quantile_from_pdf(z, model, q) for q in [0.05, 0.50, 0.95]]
    grid_q05, grid_q50, grid_q95 = [weighted_quantile_from_pdf(z, target, q) for q in [0.05, 0.50, 0.95]]

    return {
        "name": "redshift",
        "comparison_type": "rate_density_model_vs_released_grid",
        "target_weighting": target_weighting,
        "hist_l1": float(np.trapezoid(np.abs(model - target), z)),
        "model_mean": model_mean,
        "grid_mean": grid_mean,
        "mean_abs_diff": abs(model_mean - grid_mean),
        "model_q05": model_q05,
        "grid_q05": grid_q05,
        "model_q50": model_q50,
        "grid_q50": grid_q50,
        "model_q95": model_q95,
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
    parser.add_argument(
        "--z-max",
        type=float,
        default=1.9,
        help="Upper redshift for generated event samples. This is not the released redshift-grid target.",
    )
    parser.add_argument("--no-extrinsics", action="store_true")
    parser.add_argument(
        "--target-weighting",
        choices=["equal_row_pdf", "rate_weighted"],
        default="equal_row_pdf",
        help=(
            "equal_row_pdf matches fixed events per selected hyperposterior row; "
            "rate_weighted normalizes the row-averaged rate curve."
        ),
    )
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

    diagnostics = [
        compare_sample_parameter(
            samples,
            h5,
            rows,
            name,
            args.bins,
            target_weighting=args.target_weighting,
        )
        for name in SAMPLE_DISTRIBUTION_PARAMETERS
    ]
    diag = pd.DataFrame(diagnostics)
    diag_path = outdir / "joint_sampler_grid_diagnostics.csv"
    diag.to_csv(diag_path, index=False)

    redshift_diag = pd.DataFrame(
        [compare_redshift_rate_density(hyper, h5, rows, target_weighting=args.target_weighting)]
    )
    redshift_diag_path = outdir / "redshift_rate_density_diagnostics.csv"
    redshift_diag.to_csv(redshift_diag_path, index=False)

    print(f"target_weighting: {args.target_weighting}")
    print(f"wrote {samples_path}")
    print(f"wrote {diag_path}")
    print(diag.to_string(index=False))
    print(f"wrote {redshift_diag_path}")
    print(redshift_diag.to_string(index=False))


if __name__ == "__main__":
    main()
