"""Analytic one- and two-dimensional marginals for the GWTC-4 default BBH model.

The functions in this module are validation helpers. They evaluate the model
curves implied by named hyperposterior rows and are intended to be compared
against the released ``posterior/rates_on_grids`` products.

For redshift, the released grid is treated as the comoving source-frame rate
shape ``R(z)``. That is intentionally different from the detector-frame event
redshift PDF used by posterior-predictive catalog draws, which includes the
``dVc/dz / (1 + z)`` factor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift


DEFAULT_1D_PARAMETERS = (
    "mass_1",
    "mass_ratio",
    "a_1",
    "a_2",
    "cos_tilt_1",
    "cos_tilt_2",
    "redshift",
)

SAMPLE_COLUMNS = {
    "mass_1": "mass_1_source",
    "mass_ratio": "mass_ratio",
    "a_1": "a_1",
    "a_2": "a_2",
    "cos_tilt_1": "cos_tilt_1",
    "cos_tilt_2": "cos_tilt_2",
    "redshift": "redshift",
}

XLABELS = {
    "mass_1": r"$m_1\,[M_\odot]$",
    "mass_ratio": r"$q$",
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "cos_tilt_1": r"$\cos\theta_1$",
    "cos_tilt_2": r"$\cos\theta_2$",
    "redshift": r"$z$",
    "chi_eff": r"$\chi_\mathrm{eff}$",
}


@dataclass(frozen=True)
class CurveComparison:
    """Summary diagnostics for two normalized 1D curves on a common grid."""

    name: str
    l1: float
    mean_model: float
    mean_target: float
    mean_abs_diff: float
    q05_model: float
    q05_target: float
    q50_model: float
    q50_target: float
    q95_model: float
    q95_target: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "l1": self.l1,
            "mean_model": self.mean_model,
            "mean_target": self.mean_target,
            "mean_abs_diff": self.mean_abs_diff,
            "q05_model": self.q05_model,
            "q05_target": self.q05_target,
            "q50_model": self.q50_model,
            "q50_target": self.q50_target,
            "q95_model": self.q95_model,
            "q95_target": self.q95_target,
        }


def normalize_pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return ``y`` normalized as a PDF over the strictly increasing grid ``x``."""
    x = np.asarray(x, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D grid, got shape {x.shape}")
    if y.shape != x.shape:
        raise ValueError(f"Expected y shape {x.shape}, got {y.shape}")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("Grid must be strictly increasing")
    norm = np.trapezoid(y, x)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Cannot normalize PDF; integral={norm}")
    return y / norm


def normalize_pdf_2d(x: np.ndarray, y: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Normalize a 2D density whose axes are ``x`` and ``y``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    density = np.clip(np.asarray(density, dtype=float), 0.0, np.inf)
    if density.shape != (len(x), len(y)):
        raise ValueError(f"Expected density shape {(len(x), len(y))}, got {density.shape}")
    norm = np.trapezoid(np.trapezoid(density, y, axis=1), x)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Cannot normalize 2D PDF; integral={norm}")
    return density / norm


def weighted_quantile_from_pdf(x: np.ndarray, pdf: np.ndarray, q: float) -> float:
    """Quantile of a normalized or unnormalized curve on grid ``x``."""
    pdf = normalize_pdf(x, pdf)
    cdf = np.empty_like(x, dtype=float)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
    if cdf[-1] <= 0.0 or not np.isfinite(cdf[-1]):
        raise ValueError("Cannot construct CDF from PDF")
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, x))


def compare_curves(name: str, x: np.ndarray, model_pdf: np.ndarray, target_pdf: np.ndarray) -> CurveComparison:
    """Compare two 1D PDFs after normalizing both on the same grid."""
    model_pdf = normalize_pdf(x, model_pdf)
    target_pdf = normalize_pdf(x, target_pdf)
    mean_model = float(np.trapezoid(x * model_pdf, x))
    mean_target = float(np.trapezoid(x * target_pdf, x))
    q05_model, q50_model, q95_model = [weighted_quantile_from_pdf(x, model_pdf, q) for q in (0.05, 0.50, 0.95)]
    q05_target, q50_target, q95_target = [weighted_quantile_from_pdf(x, target_pdf, q) for q in (0.05, 0.50, 0.95)]
    return CurveComparison(
        name=name,
        l1=float(np.trapezoid(np.abs(model_pdf - target_pdf), x)),
        mean_model=mean_model,
        mean_target=mean_target,
        mean_abs_diff=abs(mean_model - mean_target),
        q05_model=q05_model,
        q05_target=q05_target,
        q50_model=q50_model,
        q50_target=q50_target,
        q95_model=q95_model,
        q95_target=q95_target,
    )


def row_normalized_grid_target(x: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Average per-row normalized released grid curves.

    This is the correct target for validation runs that average over selected
    hyperposterior rows with equal row weight rather than total-rate weight.
    """
    rates = np.clip(np.asarray(rates, dtype=float), 0.0, np.inf)
    if rates.ndim == 1:
        return normalize_pdf(x, rates)
    pdfs = []
    for curve in rates:
        try:
            pdfs.append(normalize_pdf(x, curve))
        except ValueError:
            continue
    if not pdfs:
        raise ValueError("No finite positive released rate curves to average")
    return normalize_pdf(x, np.mean(pdfs, axis=0))


def truncated_normal_pdf(x: np.ndarray, low: float, high: float, mu: float, sigma: float) -> np.ndarray:
    """Truncated Gaussian PDF on a supplied grid."""
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-12)
    y = np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2) / sigma
    y = np.where((x >= low) & (x <= high), y, 0.0)
    return normalize_pdf(x, y)


def spin_magnitude_pdf(row: Mapping[str, float], x: np.ndarray) -> np.ndarray:
    """PDF for either component spin magnitude in the Gaussian-component model."""
    return truncated_normal_pdf(
        x,
        low=0.0,
        high=1.0,
        mu=float(row["mu_chi"]),
        sigma=float(row["sigma_chi"]),
    )


def cos_tilt_pdf(row: Mapping[str, float], x: np.ndarray) -> np.ndarray:
    """PDF for either component ``cos_tilt`` in the NID tilt mixture."""
    xi = float(np.clip(row["xi_spin"], 0.0, 1.0))
    gaussian = truncated_normal_pdf(
        x,
        low=-1.0,
        high=1.0,
        mu=float(row["mu_spin"]),
        sigma=float(row["sigma_spin"]),
    )
    isotropic = np.where((x >= -1.0) & (x <= 1.0), 0.5, 0.0)
    return normalize_pdf(x, xi * gaussian + (1.0 - xi) * isotropic)


def redshift_rate_density_pdf(row: Mapping[str, float], z: np.ndarray) -> np.ndarray:
    """Released-grid redshift shape: source-frame rate density R(z)."""
    return normalize_pdf(z, np.power(1.0 + np.asarray(z, dtype=float), float(row["lamb"])))


def mass_ratio_marginal_pdf(
    model: GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift,
    row: Mapping[str, float],
    q_grid: np.ndarray,
    *,
    mass_1_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Compute p(q | Lambda) by integrating p(m1 | Lambda) p(q | m1, Lambda)."""
    if mass_1_grid is None:
        mass_1_grid = np.linspace(model.config.m1_min, model.config.m1_max, model.config.mass_grid_size)
    p_m1 = model.mass_sampler.marginal_mass_1_pdf(row, mass_1_grid)
    p_q_given_m1 = model.mass_sampler.conditional_mass_ratio_pdf(
        row,
        mass_1_grid,
        np.asarray(q_grid, dtype=float),
        pairwise=False,
    )
    p_q = np.trapezoid(p_m1[:, None] * p_q_given_m1, mass_1_grid, axis=0)
    return normalize_pdf(q_grid, p_q)


def mass_ratio_joint_pdf(
    model: GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift,
    row: Mapping[str, float],
    mass_1_grid: np.ndarray,
    q_grid: np.ndarray,
) -> np.ndarray:
    """Compute p(m1, q | Lambda) on a tensor product grid."""
    p_m1 = model.mass_sampler.marginal_mass_1_pdf(row, mass_1_grid)
    p_q_given_m1 = model.mass_sampler.conditional_mass_ratio_pdf(
        row,
        mass_1_grid,
        q_grid,
        pairwise=False,
    )
    return normalize_pdf_2d(mass_1_grid, q_grid, p_m1[:, None] * p_q_given_m1)


def analytic_1d_pdf_for_row(
    model: GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift,
    row: Mapping[str, float],
    name: str,
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate one named 1D model marginal for a single hyperposterior row."""
    if name == "mass_1":
        return normalize_pdf(x, model.mass_sampler.marginal_mass_1_pdf(row, x))
    if name == "mass_ratio":
        return mass_ratio_marginal_pdf(model, row, x)
    if name in {"a_1", "a_2"}:
        return spin_magnitude_pdf(row, x)
    if name in {"cos_tilt_1", "cos_tilt_2"}:
        return cos_tilt_pdf(row, x)
    if name == "redshift":
        return redshift_rate_density_pdf(row, x)
    raise KeyError(f"No analytic 1D marginal implemented for {name!r}")


def analytic_1d_pdf_for_rows(
    model: GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift,
    hyper: pd.DataFrame,
    rows: Sequence[int],
    name: str,
    x: np.ndarray,
) -> np.ndarray:
    """Equal-row average of analytic PDFs for selected hyperposterior rows."""
    pdfs = []
    for row_idx in rows:
        row = hyper.iloc[int(row_idx)].to_dict()
        pdfs.append(analytic_1d_pdf_for_row(model, row, name, x))
    if not pdfs:
        raise ValueError("No rows supplied")
    return normalize_pdf(x, np.mean(pdfs, axis=0))


def analytic_mass_ratio_joint_for_rows(
    model: GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift,
    hyper: pd.DataFrame,
    rows: Sequence[int],
    mass_1_grid: np.ndarray,
    q_grid: np.ndarray,
) -> np.ndarray:
    """Equal-row averaged analytic p(m1, q) for selected rows."""
    densities = []
    for row_idx in rows:
        row = hyper.iloc[int(row_idx)].to_dict()
        densities.append(mass_ratio_joint_pdf(model, row, mass_1_grid, q_grid))
    if not densities:
        raise ValueError("No rows supplied")
    return normalize_pdf_2d(mass_1_grid, q_grid, np.mean(densities, axis=0))


def chi_eff_from_samples(samples: pd.DataFrame) -> np.ndarray:
    """Compute chi_eff from sampled source-frame masses and aligned spin components."""
    m1 = samples["mass_1_source"].to_numpy(dtype=float)
    m2 = samples["mass_2_source"].to_numpy(dtype=float)
    a1 = samples["a_1"].to_numpy(dtype=float)
    a2 = samples["a_2"].to_numpy(dtype=float)
    c1 = samples["cos_tilt_1"].to_numpy(dtype=float)
    c2 = samples["cos_tilt_2"].to_numpy(dtype=float)
    return (m1 * a1 * c1 + m2 * a2 * c2) / (m1 + m2)
