#!/usr/bin/env python
"""Step 09: scan mass-model conventions against released mass_1 grids.

The spin/redshift projections can validate well while the mass projection fails
if we choose the wrong convention for alpha signs, Gaussian-mixture weights, or
break-mass interpretation. This script compares a small set of plausible
conventions directly against the released `posterior/rates_on_grids/mass_1`
product, without drawing Monte Carlo events.

It is a diagnostic script, not a sampler.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from popsampler.grid_rates import load_rate_grid
from popsampler.popsummary_io import get_hyperparameter_samples


@dataclass(frozen=True)
class MassConvention:
    name: str
    alpha_mode: str  # "direct" gives m^alpha, "gwpopulation" gives m^(-alpha)
    lambda_mode: str  # "absolute", "lam0_total", "lam1_total"
    break_mode: str  # "fraction_if_unit" or "absolute"
    gaussian_high_mode: str  # "fixed_100" or "mmax"


def low_mass_smoothing(x: np.ndarray, mlow: float, delta_m: float, high: float | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if delta_m <= 0:
        return ((x >= mlow) & (x <= high)).astype(float)
    y = (x - mlow) / delta_m
    out = np.zeros_like(x, dtype=float)
    out[x > high] = 0.0
    out[y >= 1.0] = 1.0
    mask = (y > 0.0) & (y < 1.0) & (x <= high)
    y_clip = np.clip(y[mask], 1e-6, 1.0 - 1e-6)
    exponent = 1.0 / y_clip - 1.0 / (1.0 - y_clip)
    out[mask] = 1.0 / (np.exp(exponent) + 1.0)
    out[x > high] = 0.0
    return out


def trapz_norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
    norm = np.trapz(y, x)
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(y)
    return y / norm


def normal_pdf(x: np.ndarray, mu: float, sigma: float, low: float, high: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma
    y *= (x >= low) & (x <= high)
    return trapz_norm(x, y)


def resolve_break(row: pd.Series, convention: MassConvention) -> float:
    value = float(row["break_mass"])
    mlow = float(row["mlow_1"])
    mmax = float(row["mmax"])
    if convention.break_mode == "fraction_if_unit" and 0.0 < value < 1.0:
        return mlow + value * (mmax - mlow)
    return float(np.clip(value, mlow, mmax))


def component_weights(row: pd.Series, convention: MassConvention) -> tuple[float, float, float]:
    lam0 = float(np.clip(row["lam_0"], 0.0, 1.0))
    lam1 = float(np.clip(row["lam_1"], 0.0, 1.0))
    if convention.lambda_mode == "absolute":
        return max(0.0, 1.0 - lam0 - lam1), lam0, lam1
    if convention.lambda_mode == "lam0_total":
        return max(0.0, 1.0 - lam0), lam0 * lam1, lam0 * (1.0 - lam1)
    if convention.lambda_mode == "lam1_total":
        return max(0.0, 1.0 - lam1), lam1 * lam0, lam1 * (1.0 - lam0)
    raise ValueError(convention.lambda_mode)


def primary_pdf(x: np.ndarray, row: pd.Series, convention: MassConvention) -> np.ndarray:
    mlow = float(row["mlow_1"])
    mmax = float(row["mmax"])
    delta_m = float(row["delta_m_1"])
    break_mass = resolve_break(row, convention)
    exp1 = float(row["alpha_1"])
    exp2 = float(row["alpha_2"])
    if convention.alpha_mode == "gwpopulation":
        exp1 = -exp1
        exp2 = -exp2
    elif convention.alpha_mode != "direct":
        raise ValueError(convention.alpha_mode)

    support = (x >= mlow) & (x <= mmax)
    continuum = np.zeros_like(x)
    below = support & (x <= break_mass)
    above = support & (x > break_mass)
    continuum[below] = np.power(np.maximum(x[below] / break_mass, 1e-300), exp1)
    continuum[above] = np.power(np.maximum(x[above] / break_mass, 1e-300), exp2)
    continuum *= low_mass_smoothing(x, mlow, delta_m, high=mmax)
    continuum = trapz_norm(x, continuum)

    gaussian_high = 100.0 if convention.gaussian_high_mode == "fixed_100" else mmax
    peak1 = normal_pdf(x, float(row["mpp_1"]), float(row["sigpp_1"]), low=mlow, high=gaussian_high)
    peak2 = normal_pdf(x, float(row["mpp_2"]), float(row["sigpp_2"]), low=mlow, high=gaussian_high)
    w_cont, w_peak1, w_peak2 = component_weights(row, convention)
    pdf = w_cont * continuum + w_peak1 * peak1 + w_peak2 * peak2
    return trapz_norm(x, pdf)


def weighted_quantile(x: np.ndarray, pdf: np.ndarray, q: float) -> float:
    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
    if cdf[-1] <= 0:
        return float("nan")
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, x))


def summarize_comparison(name: str, x: np.ndarray, estimate: np.ndarray, target: np.ndarray) -> dict[str, float | str]:
    estimate = trapz_norm(x, estimate)
    target = trapz_norm(x, target)
    return {
        "convention": name,
        "density_l1": float(np.trapz(np.abs(estimate - target), x)),
        "estimate_mean": float(np.trapz(x * estimate, x)),
        "target_mean": float(np.trapz(x * target, x)),
        "mean_abs_diff": float(abs(np.trapz(x * estimate, x) - np.trapz(x * target, x))),
        "estimate_q05": weighted_quantile(x, estimate, 0.05),
        "target_q05": weighted_quantile(x, target, 0.05),
        "estimate_q50": weighted_quantile(x, estimate, 0.50),
        "target_q50": weighted_quantile(x, target, 0.50),
        "estimate_q95": weighted_quantile(x, estimate, 0.95),
        "target_q95": weighted_quantile(x, target, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--outdir", default="validation_outputs")
    parser.add_argument("--rows", default=None, help="Optional .npy row-index file, e.g. validation_outputs/joint_sampler_rows.npy")
    parser.add_argument("--n-hyperrows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hyper = get_hyperparameter_samples(h5)
    if args.rows is not None and Path(args.rows).exists():
        rows = np.load(args.rows).astype(int)
    else:
        rng = np.random.default_rng(args.seed)
        rows = rng.choice(len(hyper), size=args.n_hyperrows, replace=False)

    grid = load_rate_grid(h5, "mass_1")
    x = grid.positions
    target = trapz_norm(x, grid.rates[rows].mean(axis=0))

    conventions = [
        MassConvention("direct_absolute_fixed100", "direct", "absolute", "fraction_if_unit", "fixed_100"),
        MassConvention("direct_lam0_total_fixed100", "direct", "lam0_total", "fraction_if_unit", "fixed_100"),
        MassConvention("direct_lam1_total_fixed100", "direct", "lam1_total", "fraction_if_unit", "fixed_100"),
        MassConvention("gwpop_absolute_fixed100", "gwpopulation", "absolute", "fraction_if_unit", "fixed_100"),
        MassConvention("gwpop_lam0_total_fixed100", "gwpopulation", "lam0_total", "fraction_if_unit", "fixed_100"),
        MassConvention("gwpop_lam1_total_fixed100", "gwpopulation", "lam1_total", "fraction_if_unit", "fixed_100"),
        MassConvention("direct_lam0_total_gmax_mmax", "direct", "lam0_total", "fraction_if_unit", "mmax"),
        MassConvention("gwpop_lam0_total_gmax_mmax", "gwpopulation", "lam0_total", "fraction_if_unit", "mmax"),
    ]

    summaries = []
    curves = {"mass_1": x, "target": target}
    for convention in conventions:
        pdfs = []
        for idx in rows:
            pdfs.append(primary_pdf(x, hyper.iloc[int(idx)], convention))
        estimate = trapz_norm(x, np.mean(pdfs, axis=0))
        summaries.append(summarize_comparison(convention.name, x, estimate, target))
        curves[convention.name] = estimate

    summary = pd.DataFrame(summaries).sort_values("density_l1")
    summary_path = outdir / "mass_convention_scan.csv"
    curves_path = outdir / "mass_convention_curves.csv"
    rows_path = outdir / "mass_convention_rows.npy"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(curves).to_csv(curves_path, index=False)
    np.save(rows_path, rows)

    print(f"rows: {len(rows)}")
    print(f"wrote {summary_path}")
    print(f"wrote {curves_path}")
    print(f"wrote {rows_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
