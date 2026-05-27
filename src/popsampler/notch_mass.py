"""Native Notch-style compact-object mass sampler.

This module intentionally does not call GWForge, gwpopulation, or any other
population-model package.  It implements a small explicit Notch-style mass
spectrum for posterior-predictive simulations inside popsampler.

The object spectrum is a continuous three-segment power law:

- NS branch:  NSmin <= m <= NSmax, slope alpha_1;
- notch/dip: NSmax <  m <  BHmin, slope alpha_dip;
- BH branch: BHmin <= m <= BHmax, slope alpha_2.

Binary masses are drawn from

    p(m1, q | Lambda) proportional to p_obj(m1) p_obj(q m1) q**beta,

with ordered components m1 >= m2, q = m2 / m1, and the same object-spectrum
support applied to both components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .samplers import InverseCDFSampler


@dataclass(frozen=True)
class NotchMassSamplerConfig:
    """Numerical settings for the native Notch mass sampler."""

    m1_min: float = 1.0
    m1_max: float = 300.0
    q_min: float = 0.001
    q_max: float = 1.0
    mass_grid_size: int = 1200
    q_grid_size: int = 500


@dataclass(frozen=True)
class NotchHyperparameters:
    """Resolved Notch hyperparameters for one hyperposterior row."""

    alpha_1: float
    alpha_dip: float
    alpha_2: float
    ns_min: float
    ns_max: float
    bh_min: float
    bh_max: float
    beta: float = 0.0


def _optional_float(row: Mapping[str, float], *names: str, default: float | None = None) -> float | None:
    for name in names:
        if name in row and not pd.isna(row[name]):
            value = float(row[name])
            if not np.isfinite(value):
                raise ValueError(f"Hyperparameter {name!r} is not finite: {value}")
            return value
    return default


def _required_float(row: Mapping[str, float], *names: str) -> float:
    value = _optional_float(row, *names, default=None)
    if value is None:
        joined = " or ".join(repr(name) for name in names)
        raise ValueError(f"Missing required Notch hyperparameter {joined}")
    return value


def resolve_notch_hyperparameters(
    row: Mapping[str, float],
    *,
    default_bh_max: float,
) -> NotchHyperparameters:
    """Resolve aliases and validate one Notch hyperposterior row."""

    hp = NotchHyperparameters(
        alpha_1=_required_float(row, "alpha_1"),
        alpha_dip=_required_float(row, "alpha_dip"),
        alpha_2=_required_float(row, "alpha_2"),
        ns_min=_required_float(row, "NSmin", "ns_min", "mmin_ns"),
        ns_max=_required_float(row, "NSmax", "ns_max", "mmax_ns"),
        bh_min=_required_float(row, "BHmin", "bh_min", "mmin_bh"),
        bh_max=_optional_float(row, "BHmax", "bh_max", "mmax", default=default_bh_max),
        beta=_optional_float(row, "beta", default=0.0),
    )
    validate_notch_hyperparameters(hp)
    return hp


def validate_notch_hyperparameters(hp: NotchHyperparameters) -> None:
    """Check the physical ordering of the Notch support."""

    if not (0.0 < hp.ns_min < hp.ns_max <= hp.bh_min < hp.bh_max):
        raise ValueError(
            "Expected Notch mass bounds to obey "
            "0 < NSmin < NSmax <= BHmin < BHmax/mmax; got "
            f"NSmin={hp.ns_min}, NSmax={hp.ns_max}, BHmin={hp.bh_min}, BHmax={hp.bh_max}"
        )


def object_mass_pdf(mass: np.ndarray, hp: NotchHyperparameters) -> np.ndarray:
    """Unnormalized continuous Notch object-mass spectrum p_obj(m)."""

    m = np.asarray(mass, dtype=float)
    pdf = np.zeros_like(m, dtype=float)

    ns = (m >= hp.ns_min) & (m <= hp.ns_max)
    dip = (m > hp.ns_max) & (m < hp.bh_min)
    bh = (m >= hp.bh_min) & (m <= hp.bh_max)

    # Normalize the branch amplitudes relative to NSmax so that the piecewise
    # curve is continuous at NSmax and BHmin. The final normalization is done by
    # the inverse-CDF samplers.
    pdf[ns] = np.power(np.maximum(m[ns], 1.0e-300) / hp.ns_max, hp.alpha_1)
    pdf[dip] = np.power(np.maximum(m[dip], 1.0e-300) / hp.ns_max, hp.alpha_dip)
    bh_prefactor = np.power(hp.bh_min / hp.ns_max, hp.alpha_dip)
    pdf[bh] = bh_prefactor * np.power(np.maximum(m[bh], 1.0e-300) / hp.bh_min, hp.alpha_2)
    return np.clip(pdf, 0.0, np.inf)


class NotchMassSampler:
    """Sample ordered binary masses from the native Notch object spectrum."""

    def __init__(self, config: NotchMassSamplerConfig | None = None):
        self.config = NotchMassSamplerConfig() if config is None else config

    def sample(
        self,
        row: Mapping[str, float],
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng() if rng is None else rng
        hp = resolve_notch_hyperparameters(row, default_bh_max=self.config.m1_max)

        m1_lo = max(self.config.m1_min, hp.ns_min)
        m1_hi = min(self.config.m1_max, hp.bh_max)
        if not (m1_lo < m1_hi):
            raise ValueError(f"Empty Notch m1 support: [{m1_lo}, {m1_hi}]")

        m1_grid = np.linspace(m1_lo, m1_hi, self.config.mass_grid_size)
        q_grid = np.linspace(self.config.q_min, self.config.q_max, self.config.q_grid_size)
        p_m1 = self._marginal_m1_pdf(m1_grid, q_grid, hp)
        m1 = InverseCDFSampler.from_pdf(m1_grid, p_m1).sample(n, rng)

        q = np.empty(n)
        for i, this_m1 in enumerate(m1):
            q_pdf = self._conditional_q_pdf(q_grid, float(this_m1), hp)
            q[i] = InverseCDFSampler.from_pdf(q_grid, q_pdf).sample(1, rng)[0]

        m2 = q * m1
        return {
            "mass_1": m1,
            "mass_2": m2,
            "mass_ratio": q,
            "mass_sampler_mode": np.full(n, "native_notch_object_spectrum_q_power_pairing", dtype=object),
        }

    def _marginal_m1_pdf(
        self,
        m1_grid: np.ndarray,
        q_grid: np.ndarray,
        hp: NotchHyperparameters,
    ) -> np.ndarray:
        p1 = object_mass_pdf(m1_grid, hp)
        pdf = np.zeros_like(m1_grid)
        for i, m1 in enumerate(m1_grid):
            q_pdf = self._conditional_q_pdf(q_grid, float(m1), hp, p1_value=float(p1[i]))
            pdf[i] = np.trapz(q_pdf, q_grid)
        return np.clip(pdf, 0.0, np.inf)

    def _conditional_q_pdf(
        self,
        q_grid: np.ndarray,
        m1: float,
        hp: NotchHyperparameters,
        *,
        p1_value: float | None = None,
    ) -> np.ndarray:
        q = np.asarray(q_grid, dtype=float)
        m2 = q * float(m1)
        p1 = float(object_mass_pdf(np.asarray([m1]), hp)[0]) if p1_value is None else p1_value
        p2 = object_mass_pdf(m2, hp)
        valid = (q >= self.config.q_min) & (q <= self.config.q_max) & (m2 >= hp.ns_min) & (m2 <= m1)
        pdf = np.where(valid, p1 * p2 * np.power(np.maximum(q, 1.0e-300), hp.beta), 0.0)
        return np.clip(pdf, 0.0, np.inf)
