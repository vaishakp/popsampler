#!/usr/bin/env python
"""Step 12: deterministic redshift convention scan against released grids.

The GWTC-4 paper defines the Power Law Redshift model through the comoving
source-frame merger-rate density, R(z) ∝ (1 + z)^lamb. Depending on the data
product, a released 1D redshift grid can represent either the comoving rate
density R(z), a source-frame differential rate dR/dz, or a detector-frame event
counting distribution proportional to dVc/dz/(1+z). This script compares these
conventions directly against the released redshift grid, row by row, without
Monte Carlo event-sampling noise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from popsampler.cosmology import DEFAULT_COSMOLOGY, differential_comoving_volume_full_sky
from popsampler.grid_rates import load_rate_grid
from popsampler.popsummary_io import get_hyperparameter_samples


@dataclass(frozen=True)
class RedshiftConvention:
    name: str
    volume_power: float
    time_dilation_power: float
    lambda_shift: float


def trapz_normalize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
    norm = np.trapezoid(y, x)
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(y)
    return y / norm


def redshift_pdf(
    z: np.ndarray,
    lamb: float,
    convention: RedshiftConvention,
    *,
    cosmology=DEFAULT_COSMOLOGY,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    pdf = np.power(1.0 + z, float(lamb) + convention.lambda_shift)
    if convention.volume_power != 0.0:
        dvc_dz = differential_comoving_volume_full_sky(z, cosmology=cosmology)
        pdf *= np.power(np.clip(dvc_dz, 0.0, np.inf), convention.volume_power)
    if convention.time_dilation_power != 0.0:
        pdf *= np.power(1.0 + z, convention.time_dilation_power)
    return trapz_normalize(z, pdf)


def target_pdf(x: np.ndarray, rates: np.ndarray, *, target_weighting: str) -> np.ndarray:
    rates = np.clip(np.asarray(rates, dtype=float), 0.0, np.inf)
    if target_weighting == "rate_weighted":
        return trapz_normalize(x, rates.mean(axis=0))
    if target_weighting == "equal_row_pdf":
        norms = np.trapezoid(rates, x, axis=1)
        good = np.isfinite(norms) & (norms > 0)
        if not np.any(good):
            raise ValueError("No selected redshift-grid rows have positive finite integrals")
        pdfs = rates[good] / norms[good, None]
        return trapz_normalize(x, pdfs.mean(axis=0))
    raise ValueError(target_weighting)


def weighted_quantile(x: np.ndarray, pdf: np.ndarray, q: float) -> float:
    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
    if cdf[-1] <= 0:
        return float("nan")
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, x))


def summarize(
    z: np.ndarray,
    estimate: np.ndarray,
    target: np.ndarray,
    convention: RedshiftConvention,
    *,
    target_weighting: str,
) -> dict[str, float | str]:
    estimate = trapz_normalize(z, estimate)
    target = trapz_normalize(z, target)
    estimate_mean = float(np.trapezoid(z * estimate, z))
    target_mean = float(np.trapezoid(z * target, z))
    return {
        "convention": convention.name,
        "target_weighting": target_weighting,
        "volume_power": convention.volume_power,
        "time_dilation_power": convention.time_dilation_power,
        "lambda_shift": convention.lambda_shift,
        "l1": float(np.trapezoid(np.abs(estimate - target), z)),
        "estimate_mean": estimate_mean,
        "target_mean": target_mean,
        "mean_abs_diff": abs(estimate_mean - target_mean),
        "estimate_q05": weighted_quantile(z, estimate, 0.05),
        "target_q05": weighted_quantile(z, target, 0.05),
        "estimate_q50": weighted_quantile(z, estimate, 0.50),
        "target_q50": weighted_quantile(z, target, 0.50),
        "estimate_q95": weighted_quantile(z, estimate, 0.95),
        "target_q95": weighted_quantile(z, target, 0.95),
    }


def build_conventions() -> list[RedshiftConvention]:
    conventions = [
        # Paper-defined comoving source-frame rate density R(z) ∝ (1+z)^lamb.
        RedshiftConvention("comoving_rate_density", volume_power=0.0, time_dilation_power=0.0, lambda_shift=0.0),
        # Source-frame differential rate dR/dz ∝ R(z) dVc/dz.
        RedshiftConvention("source_frame_dR_dz", volume_power=1.0, time_dilation_power=0.0, lambda_shift=0.0),
        # Detector-frame event counting distribution dN/dt_det dz.
        RedshiftConvention("detector_frame_dN_dz", volume_power=1.0, time_dilation_power=-1.0, lambda_shift=0.0),
        # No volume, but with time dilation only.
        RedshiftConvention("rate_density_time_dilated", volume_power=0.0, time_dilation_power=-1.0, lambda_shift=0.0),
    ]
    # Scan near-by exponent conventions, useful for catching sign/offset mistakes
    # or a released grid stored in an implicitly transformed coordinate.
    for volume_power in [0.0, 1.0]:
        for time_dilation_power in [0.0, -1.0]:
            for lambda_shift in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
                name = f"vol{volume_power:g}__td{time_dilation_power:g}__lshift{lambda_shift:g}"
                conventions.append(
                    RedshiftConvention(
                        name=name,
                        volume_power=volume_power,
                        time_dilation_power=time_dilation_power,
                        lambda_shift=lambda_shift,
                    )
                )
    # Deduplicate by parameter tuple while preserving order.
    unique: dict[tuple[float, float, float], RedshiftConvention] = {}
    for convention in conventions:
        unique.setdefault(
            (convention.volume_power, convention.time_dilation_power, convention.lambda_shift),
            convention,
        )
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--outdir", default="validation_outputs")
    parser.add_argument("--rows", default="validation_outputs/joint_sampler_rows.npy")
    parser.add_argument("--n-hyperrows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--target-weighting",
        choices=["equal_row_pdf", "rate_weighted"],
        default="equal_row_pdf",
        help="Use equal_row_pdf for fixed events_per_row validation; rate_weighted for rate-weighted catalog ensembles.",
    )
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hyper = get_hyperparameter_samples(h5)
    rows_path = Path(args.rows)
    if rows_path.exists():
        rows = np.load(rows_path).astype(int)
    else:
        rng = np.random.default_rng(args.seed)
        rows = rng.choice(len(hyper), size=args.n_hyperrows, replace=False)

    grid = load_rate_grid(h5, "redshift")
    z = grid.positions
    target = target_pdf(z, grid.rates[rows], target_weighting=args.target_weighting)

    summaries = []
    curves = {"redshift": z, f"target_{args.target_weighting}": target}
    for convention in build_conventions():
        pdfs = []
        for idx in rows:
            row = hyper.iloc[int(idx)]
            pdfs.append(redshift_pdf(z, float(row["lamb"]), convention))
        estimate = trapz_normalize(z, np.mean(pdfs, axis=0))
        summaries.append(summarize(z, estimate, target, convention, target_weighting=args.target_weighting))
        curves[convention.name] = estimate

    summary = pd.DataFrame(summaries).sort_values(["l1", "mean_abs_diff"])
    summary_path = outdir / "redshift_convention_scan.csv"
    curves_path = outdir / "redshift_convention_curves.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(curves).to_csv(curves_path, index=False)

    print(f"rows: {len(rows)}")
    print(f"target_weighting: {args.target_weighting}")
    print(f"wrote {summary_path}")
    print(f"wrote {curves_path}")
    print(summary.head(30).to_string(index=False))
    print("\nBest by mean_abs_diff:")
    print(summary.sort_values(["mean_abs_diff", "l1"]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
