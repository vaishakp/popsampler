"""GWTC-4 BBH mass sampling helpers backed by gwpopulation conventions.

The GWTC-4 release model names and figure scripts point to gwpopulation mass
models. This module uses gwpopulation's mass-model functions for the PDF
conventions and adds only the inverse-CDF sampling layer needed by popsampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .samplers import InverseCDFSampler


class GWPopulationDependencyError(RuntimeError):
    """Raised when gwpopulation is required but unavailable."""


@dataclass(frozen=True)
class GWTCPeakBrokenPowerLawMassConfig:
    """Numerical settings for GWTC-style mass sampling."""

    m1_min: float = 2.0
    m1_max: float = 300.0
    q_min: float = 0.001
    q_max: float = 1.0
    mass_grid_size: int = 4096
    q_grid_size: int = 2048
    gaussian_mass_maximum: float = 100.0


def _required(row: Mapping[str, float], name: str) -> float:
    if name not in row or pd.isna(row[name]):
        raise ValueError(f"Missing required hyperparameter {name!r}")
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"Hyperparameter {name!r} is not finite: {value}")
    return value


def _import_gwpopulation_mass():
    try:
        from gwpopulation.models import mass as gwmass
    except Exception as exc:  # pragma: no cover - optional dependency
        raise GWPopulationDependencyError(
            "GWTC-4 mass sampling requires gwpopulation. Install with "
            "`python -m pip install -e '.[gwtc4]'` or install gwpopulation manually."
        ) from exc
    return gwmass


def _trapz_normalize(x: np.ndarray, y: np.ndarray, *, label: str) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
    norm = np.trapz(y, x)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Could not normalize {label}: integral={norm}")
    return y / norm


def _normal_pdf_fallback(x: np.ndarray, mu: float, sigma: float, low: float, high: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma
    y *= (x >= low) & (x <= high)
    return _trapz_normalize(x, y, label="fallback truncated Gaussian")


def _gw_truncnorm(gwmass, x: np.ndarray, *, mu: float, sigma: float, low: float, high: float) -> np.ndarray:
    # gwpopulation imports truncnorm into gwpopulation.models.mass from gwpopulation.utils.
    if hasattr(gwmass, "truncnorm"):
        return np.asarray(gwmass.truncnorm(x, mu=mu, sigma=sigma, low=low, high=high), dtype=float)
    return _normal_pdf_fallback(x, mu=mu, sigma=sigma, low=low, high=high)


def _gw_smoothing(gwmass, x: np.ndarray, *, mmin: float, mmax: float | np.ndarray, delta_m: float) -> np.ndarray:
    return np.asarray(
        gwmass.BaseSmoothedMassDistribution.smoothing(x, mmin=mmin, mmax=mmax, delta_m=delta_m),
        dtype=float,
    )


def primary_mass_pdf_gwpopulation(
    mass_1: np.ndarray,
    row: Mapping[str, float],
    *,
    config: GWTCPeakBrokenPowerLawMassConfig | None = None,
) -> np.ndarray:
    """Evaluate the GWTC-4 primary-mass PDF using gwpopulation conventions.

    This follows the observed GWTC-4 parameter naming:

    - ``alpha_1, alpha_2, break_mass`` are passed to gwpopulation's broken
      power-law primary-mass function.
    - ``lam_0`` is the total two-Gaussian fraction.
    - ``lam_1`` is the lower-peak subfraction inside that total Gaussian weight.
    - low-mass smoothing is gwpopulation's ``BaseSmoothedMassDistribution`` taper.
    """
    cfg = GWTCPeakBrokenPowerLawMassConfig() if config is None else config
    gwmass = _import_gwpopulation_mass()
    x = np.asarray(mass_1, dtype=float)

    mlow_1 = _required(row, "mlow_1")
    mmax = _required(row, "mmax")
    delta_m_1 = _required(row, "delta_m_1")
    alpha_1 = _required(row, "alpha_1")
    alpha_2 = _required(row, "alpha_2")
    break_fraction = _required(row, "break_mass")
    lam = float(np.clip(_required(row, "lam_0"), 0.0, 1.0))
    lam_1 = float(np.clip(_required(row, "lam_1"), 0.0, 1.0))

    # gwpopulation's double_power_law_primary_mass interprets break_fraction as
    # the fraction of the interval between mmin and mmax.
    p_pow = np.asarray(
        gwmass.double_power_law_primary_mass(
            mass=x,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mlow_1,
            mmax=mmax,
            break_fraction=break_fraction,
        ),
        dtype=float,
    )
    p_pow *= _gw_smoothing(gwmass, x, mmin=mlow_1, mmax=cfg.m1_max, delta_m=delta_m_1)
    p_pow = _trapz_normalize(x, p_pow, label="gwpopulation broken-power-law primary mass")

    high = cfg.gaussian_mass_maximum
    p_norm1 = _gw_truncnorm(
        gwmass,
        x,
        mu=_required(row, "mpp_1"),
        sigma=_required(row, "sigpp_1"),
        low=mlow_1,
        high=high,
    )
    p_norm2 = _gw_truncnorm(
        gwmass,
        x,
        mu=_required(row, "mpp_2"),
        sigma=_required(row, "sigpp_2"),
        low=mlow_1,
        high=high,
    )
    p_norm1 = _trapz_normalize(x, p_norm1, label="gwpopulation lower Gaussian peak")
    p_norm2 = _trapz_normalize(x, p_norm2, label="gwpopulation upper Gaussian peak")

    pdf = (1.0 - lam) * p_pow + lam * lam_1 * p_norm1 + lam * (1.0 - lam_1) * p_norm2
    return _trapz_normalize(x, pdf, label="GWTC-4 primary mass mixture")


def conditional_mass_ratio_pdf_gwpopulation(
    q: np.ndarray,
    mass_1: float,
    row: Mapping[str, float],
) -> np.ndarray:
    """Evaluate p(q | mass_1, Lambda) using gwpopulation conventions."""
    gwmass = _import_gwpopulation_mass()
    q = np.asarray(q, dtype=float)
    mlow_2 = _required(row, "mlow_2")
    delta_m_2 = _required(row, "delta_m_2")
    beta = _required(row, "beta")

    if hasattr(gwmass, "powerlaw"):
        pdf = np.asarray(gwmass.powerlaw(q, beta, high=1.0, low=mlow_2 / mass_1), dtype=float)
    else:  # pragma: no cover - defensive for old gwpopulation versions
        pdf = np.where(q >= mlow_2 / mass_1, np.power(np.maximum(q, 1e-300), beta), 0.0)
    pdf *= _gw_smoothing(gwmass, q * mass_1, mmin=mlow_2, mmax=mass_1, delta_m=delta_m_2)
    return _trapz_normalize(q, pdf, label="gwpopulation conditional mass-ratio")


def sample_masses_gwpopulation(
    row: Mapping[str, float],
    n: int,
    *,
    rng: np.random.Generator,
    config: GWTCPeakBrokenPowerLawMassConfig | None = None,
) -> dict[str, np.ndarray]:
    """Draw masses from the GWTC-4 mass model using gwpopulation PDF conventions."""
    cfg = GWTCPeakBrokenPowerLawMassConfig() if config is None else config
    mlow_1 = _required(row, "mlow_1")
    mlow_2 = _required(row, "mlow_2")
    mmax = _required(row, "mmax")

    lo = max(cfg.m1_min, min(mlow_1, mlow_2) * 0.8)
    hi = min(cfg.m1_max, max(mmax, cfg.gaussian_mass_maximum))
    if lo >= hi:
        raise ValueError(f"Invalid m1 grid limits: lo={lo}, hi={hi}")

    m1_grid = np.linspace(lo, hi, cfg.mass_grid_size)
    p_m1 = primary_mass_pdf_gwpopulation(m1_grid, row, config=cfg)
    m1 = InverseCDFSampler.from_pdf(m1_grid, p_m1).sample(n, rng)

    q = np.empty(n)
    q_grid_base = np.linspace(cfg.q_min, cfg.q_max, cfg.q_grid_size)
    for i, mass_1 in enumerate(m1):
        q_min = max(cfg.q_min, mlow_2 / mass_1)
        q_grid = q_grid_base[q_grid_base >= q_min]
        if len(q_grid) < 2:
            q[i] = min(1.0, max(q_min, cfg.q_min))
            continue
        p_q = conditional_mass_ratio_pdf_gwpopulation(q_grid, float(mass_1), row)
        q[i] = InverseCDFSampler.from_pdf(q_grid, p_q).sample(1, rng)[0]

    m2 = q * m1
    chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    return {
        "mass_1_source": m1,
        "mass_2_source": m2,
        "mass_ratio": q,
        "chirp_mass_source": chirp,
        "total_mass_source": m1 + m2,
        "mass_sampler_backend": np.full(n, "gwpopulation", dtype=object),
    }
