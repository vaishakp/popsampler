#!/usr/bin/env python
"""Step 11: deterministic primary-mass convention scan against released grids.

The stochastic sampler validation shows that spin projections match the released
1D grids while mass_1 remains too heavy. This script removes Monte Carlo noise by
comparing model PDFs directly on the released mass_1 grid.

It scans the conventions most likely to matter for the GWTC-4 two-peak broken
power-law mass model:

* effective power-law mmax: row mmax, fixed 100, or the released grid maximum;
* Gaussian peak support: fixed 100, effective mmax, or grid maximum;
* peak weights: lam_0 as total Gaussian fraction vs absolute lam_0/lam_1;
* whether the primary low-mass smoothing also multiplies Gaussian peaks.

The target is the equal-row average of per-row normalized released rate grids,
which matches validation runs with fixed events_per_row.
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
class PrimaryMassConvention:
    name: str
    mmax_mode: str
    peak_high_mode: str
    lambda_mode: str
    smooth_peaks: bool


def trapz_normalize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
    norm = np.trapezoid(y, x)
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(y)
    return y / norm


def low_mass_smoothing(x: np.ndarray, low: float, delta_m: float, high: float | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    high = np.asarray(high, dtype=float)
    if delta_m <= 0:
        return np.where((x >= low) & (x <= high), 1.0, 0.0)
    y = (x - low) / delta_m
    out = np.zeros_like(x, dtype=float)
    out[y >= 1.0] = 1.0
    mask = (y > 0.0) & (y < 1.0)
    y_clip = np.clip(y[mask], 1e-12, 1.0 - 1e-12)
    exponent = np.clip(1.0 / y_clip + 1.0 / (y_clip - 1.0), -700.0, 700.0)
    out[mask] = 1.0 / (np.exp(exponent) + 1.0)
    return np.where(x <= high, out, 0.0)


def normal_component(x: np.ndarray, *, mu: float, sigma: float, low: float, high: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma
    return np.where((x >= low) & (x <= high), y, 0.0)


def resolve_mmax(row: pd.Series, x: np.ndarray, mode: str) -> float:
    if mode == "row_mmax":
        return float(row["mmax"])
    if mode == "fixed_100":
        return 100.0
    if mode == "grid_max":
        return float(np.max(x))
    raise ValueError(mode)


def resolve_peak_high(effective_mmax: float, x: np.ndarray, mode: str) -> float:
    if mode == "fixed_100":
        return 100.0
    if mode == "effective_mmax":
        return float(effective_mmax)
    if mode == "grid_max":
        return float(np.max(x))
    raise ValueError(mode)


def resolve_weights(row: pd.Series, mode: str) -> tuple[float, float, float]:
    lam0 = float(np.clip(row["lam_0"], 0.0, 1.0))
    lam1 = float(np.clip(row["lam_1"], 0.0, 1.0))
    if mode == "lam0_total_lam1_lower":
        weights = (1.0 - lam0, lam0 * lam1, lam0 * (1.0 - lam1))
    elif mode == "lam0_total_lam1_upper":
        weights = (1.0 - lam0, lam0 * (1.0 - lam1), lam0 * lam1)
    elif mode == "lam1_total_lam0_lower":
        weights = (1.0 - lam1, lam1 * lam0, lam1 * (1.0 - lam0))
    elif mode == "absolute_lam0_lam1":
        weights = (max(0.0, 1.0 - lam0 - lam1), lam0, lam1)
    else:
        raise ValueError(mode)
    total = sum(weights)
    if not np.isfinite(total) or total <= 0:
        return (1.0, 0.0, 0.0)
    return tuple(float(w / total) for w in weights)


def primary_pdf(x: np.ndarray, row: pd.Series, convention: PrimaryMassConvention, gwpop_mass) -> np.ndarray:
    mmin = float(row["mlow_1"])
    effective_mmax = resolve_mmax(row, x, convention.mmax_mode)
    effective_mmax = max(effective_mmax, mmin + 1e-6)
    break_mass = float(row["break_mass"])
    if 0.0 < break_mass < 1.0:
        break_fraction = break_mass
    else:
        break_fraction = (break_mass - mmin) / (effective_mmax - mmin)
    break_fraction = float(np.clip(break_fraction, 0.0, 1.0))
    delta_m = float(row["delta_m_1"])
    smooth = low_mass_smoothing(x, mmin, delta_m, high=effective_mmax)

    continuum = gwpop_mass.double_power_law_primary_mass(
        x,
        alpha_1=float(row["alpha_1"]),
        alpha_2=float(row["alpha_2"]),
        mmin=mmin,
        mmax=effective_mmax,
        break_fraction=break_fraction,
    ) * smooth

    peak_high = resolve_peak_high(effective_mmax, x, convention.peak_high_mode)
    lower_peak = normal_component(
        x,
        mu=float(row["mpp_1"]),
        sigma=float(row["sigpp_1"]),
        low=mmin,
        high=peak_high,
    )
    upper_peak = normal_component(
        x,
        mu=float(row["mpp_2"]),
        sigma=float(row["sigpp_2"]),
        low=mmin,
        high=peak_high,
    )
    if convention.smooth_peaks:
        lower_peak *= smooth
        upper_peak *= smooth

    continuum = trapz_normalize(x, continuum)
    lower_peak = trapz_normalize(x, lower_peak)
    upper_peak = trapz_normalize(x, upper_peak)
    w_cont, w_low, w_high = resolve_weights(row, convention.lambda_mode)
    return trapz_normalize(x, w_cont * continuum + w_low * lower_peak + w_high * upper_peak)


def target_pdf(x: np.ndarray, rates: np.ndarray) -> np.ndarray:
    rates = np.clip(np.asarray(rates, dtype=float), 0.0, np.inf)
    norms = np.trapezoid(rates, x, axis=1)
    good = np.isfinite(norms) & (norms > 0)
    pdfs = rates[good] / norms[good, None]
    return trapz_normalize(x, pdfs.mean(axis=0))


def weighted_quantile(x: np.ndarray, pdf: np.ndarray, q: float) -> float:
    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
    if cdf[-1] <= 0:
        return float("nan")
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, x))


def summarize(x: np.ndarray, estimate: np.ndarray, target: np.ndarray, convention: PrimaryMassConvention) -> dict[str, float | str | bool]:
    estimate = trapz_normalize(x, estimate)
    target = trapz_normalize(x, target)
    return {
        "convention": convention.name,
        "mmax_mode": convention.mmax_mode,
        "peak_high_mode": convention.peak_high_mode,
        "lambda_mode": convention.lambda_mode,
        "smooth_peaks": convention.smooth_peaks,
        "l1": float(np.trapezoid(np.abs(estimate - target), x)),
        "estimate_mean": float(np.trapezoid(x * estimate, x)),
        "target_mean": float(np.trapezoid(x * target, x)),
        "mean_abs_diff": float(abs(np.trapezoid(x * estimate, x) - np.trapezoid(x * target, x))),
        "estimate_q05": weighted_quantile(x, estimate, 0.05),
        "target_q05": weighted_quantile(x, target, 0.05),
        "estimate_q50": weighted_quantile(x, estimate, 0.50),
        "target_q50": weighted_quantile(x, target, 0.50),
        "estimate_q95": weighted_quantile(x, estimate, 0.95),
        "target_q95": weighted_quantile(x, target, 0.95),
    }


def build_conventions() -> list[PrimaryMassConvention]:
    conventions = []
    for mmax_mode in ["row_mmax", "fixed_100", "grid_max"]:
        for peak_high_mode in ["fixed_100", "effective_mmax", "grid_max"]:
            for lambda_mode in [
                "lam0_total_lam1_lower",
                "lam0_total_lam1_upper",
                "lam1_total_lam0_lower",
                "absolute_lam0_lam1",
            ]:
                for smooth_peaks in [True, False]:
                    name = f"{mmax_mode}__{peak_high_mode}__{lambda_mode}__smooth{int(smooth_peaks)}"
                    conventions.append(
                        PrimaryMassConvention(
                            name=name,
                            mmax_mode=mmax_mode,
                            peak_high_mode=peak_high_mode,
                            lambda_mode=lambda_mode,
                            smooth_peaks=smooth_peaks,
                        )
                    )
    return conventions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--outdir", default="validation_outputs")
    parser.add_argument("--rows", default="validation_outputs/joint_sampler_rows.npy")
    parser.add_argument("--n-hyperrows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import gwpopulation.models.mass as gwpop_mass

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

    grid = load_rate_grid(h5, "mass_1")
    x = grid.positions
    target = target_pdf(x, grid.rates[rows])

    summaries = []
    curves = {"mass_1": x, "target_equal_row_pdf": target}
    for convention in build_conventions():
        pdfs = []
        for idx in rows:
            pdfs.append(primary_pdf(x, hyper.iloc[int(idx)], convention, gwpop_mass))
        estimate = trapz_normalize(x, np.mean(pdfs, axis=0))
        summaries.append(summarize(x, estimate, target, convention))
        curves[convention.name] = estimate

    summary = pd.DataFrame(summaries).sort_values(["mean_abs_diff", "l1"])
    summary_path = outdir / "primary_mass_convention_scan.csv"
    curves_path = outdir / "primary_mass_convention_curves.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(curves).to_csv(curves_path, index=False)

    print(f"rows: {len(rows)}")
    print(f"wrote {summary_path}")
    print(f"wrote {curves_path}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
