"""Model-specific posterior-predictive BBH samplers.

This module deliberately separates two tasks:

1. draw a coherent hyperparameter row Λ from the released hyperposterior;
2. draw source parameters θ from the population model conditional on Λ.

The first implemented target is the GWTC-4.0 strongly modeled BBH file
``BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5``.

Unlike the validation-only rate-grid path, this sampler draws masses
conditionally from a single named hyperposterior row and therefore preserves the
model-level conditional structure that is needed for catalog generation. Its 1D
projections still need to be validated against the released ``rates_on_grids``
products before publication use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import warnings

import numpy as np
import pandas as pd

from .cosmology import DEFAULT_COSMOLOGY, luminosity_distance_from_redshift, power_law_redshift_pdf
from .extrinsics import sample_isotropic_extrinsics
from .samplers import InverseCDFSampler


class ModelValidationError(ValueError):
    """Raised when a hyperposterior row cannot be mapped to a model."""


def _required(row: Mapping[str, float], name: str) -> float:
    if name not in row or pd.isna(row[name]):
        raise ModelValidationError(f"Missing required hyperparameter {name!r}")
    value = float(row[name])
    if not np.isfinite(value):
        raise ModelValidationError(f"Hyperparameter {name!r} is not finite: {value}")
    return value


def _optional(row: Mapping[str, float], name: str, default: float) -> float:
    if name not in row or pd.isna(row[name]):
        return float(default)
    value = float(row[name])
    if not np.isfinite(value):
        return float(default)
    return value


def _trapz_normalize(x: np.ndarray, y: np.ndarray, *, label: str) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
    norm = np.trapz(y, x)
    if not np.isfinite(norm) or norm <= 0:
        raise ModelValidationError(f"Could not normalize {label}: integral={norm}")
    return y / norm


@dataclass
class BBHDefaultModelConfig:
    """Numerical settings for the GWTC-4 default BBH sampler.

    ``z_max`` defaults to a CE-useful value rather than the released LVK grid
    maximum. For validation against ``rates_on_grids/redshift`` use the grid's
    own support, e.g. ``z_max=1.9``.
    """

    m1_min: float = 2.0
    m1_max: float = 300.0
    q_min: float = 0.001
    q_max: float = 1.0
    z_min: float = 1.0e-6
    z_max: float = 20.0
    mass_grid_size: int = 4096
    q_grid_size: int = 2048
    z_grid_size: int = 4096
    spin_grid_size: int = 2048
    cosmology: object = field(default_factory=lambda: DEFAULT_COSMOLOGY)
    warn_if_unvalidated: bool = True


class GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift:
    """Joint sampler for the GWTC-4 named default BBH model.

    Required named hyperparameters are the ones exposed by the GWTC-4
    ``popsummary`` metadata:

    masses:
        ``alpha_1, alpha_2, beta, break_mass, delta_m_1, delta_m_2,``
        ``lam_0, lam_1, mlow_1, mlow_2, mmax, mpp_1, mpp_2,``
        ``sigpp_1, sigpp_2``

    spins:
        ``mu_chi, sigma_chi, mu_spin, sigma_spin, xi_spin``. The file also
        contains ``alpha_chi, beta_chi, amax`` columns, but the Gaussian-component
        sampler convention uses the five parameters above for generation.

    redshift/rate:
        ``lamb`` and optionally ``rate`` / ``log_10_rate``.

    Notes
    -----
    The mass sampler uses the model's conditional factorization: it first draws
    ``mass_1`` from a smoothed broken power law plus two Gaussian peaks, then
    draws ``mass_ratio`` conditionally on ``mass_1`` through a power law with a
    secondary-mass low-mass smoothing factor. This is the critical improvement
    over independent 1D marginal sampling.

    The spin sampler follows the Gaussian-component convention used by the
    release-style simulation code: spin magnitudes are independent truncated
    Gaussians on ``[0, 1]`` with ``mu_chi`` and ``sigma_chi``; each binary draws a
    single formation channel with probability ``xi_spin``. In the Gaussian
    channel, both spin tilts are drawn from a truncated Gaussian on ``[-1, 1]``
    with ``mu_spin`` and ``sigma_spin``. In the isotropic/dynamical channel, both
    spin tilts are uniform on ``[-1, 1]``.
    """

    name = "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift"

    MASS_PARAMETERS = (
        "alpha_1",
        "alpha_2",
        "beta",
        "break_mass",
        "delta_m_1",
        "delta_m_2",
        "lam_0",
        "lam_1",
        "mlow_1",
        "mlow_2",
        "mmax",
        "mpp_1",
        "mpp_2",
        "sigpp_1",
        "sigpp_2",
    )
    SPIN_PARAMETERS = (
        "mu_chi",
        "sigma_chi",
        "mu_spin",
        "sigma_spin",
        "xi_spin",
    )
    REDSHIFT_PARAMETERS = ("lamb",)
    REQUIRED_PARAMETERS = MASS_PARAMETERS + SPIN_PARAMETERS + REDSHIFT_PARAMETERS

    def __init__(self, config: BBHDefaultModelConfig | None = None):
        self.config = BBHDefaultModelConfig() if config is None else config
        self._warned = False

    def validate_hyperparameters(self, row: Mapping[str, float]) -> None:
        missing = [name for name in self.REQUIRED_PARAMETERS if name not in row or pd.isna(row[name])]
        if missing:
            raise ModelValidationError(f"Missing required hyperparameters: {missing}")
        for name in self.REQUIRED_PARAMETERS:
            _required(row, name)

        mlow_1 = _required(row, "mlow_1")
        mlow_2 = _required(row, "mlow_2")
        mmax = _required(row, "mmax")
        if not (0 < mlow_1 < mmax):
            raise ModelValidationError(f"Expected 0 < mlow_1 < mmax, got {mlow_1}, {mmax}")
        if not (0 < mlow_2 < mmax):
            raise ModelValidationError(f"Expected 0 < mlow_2 < mmax, got {mlow_2}, {mmax}")
        if _required(row, "sigpp_1") <= 0 or _required(row, "sigpp_2") <= 0:
            raise ModelValidationError("Gaussian peak widths sigpp_1/sigpp_2 must be positive")
        if _required(row, "sigma_chi") <= 0 or _required(row, "sigma_spin") <= 0:
            raise ModelValidationError("Gaussian spin parameters sigma_chi/sigma_spin must be positive")
        xi = _required(row, "xi_spin")
        if not (0.0 <= xi <= 1.0):
            raise ModelValidationError(f"Expected 0 <= xi_spin <= 1, got {xi}")

    def sample(
        self,
        hyper_row: Mapping[str, float],
        n: int,
        *,
        rng: np.random.Generator | None = None,
        include_extrinsics: bool = True,
    ) -> pd.DataFrame:
        """Draw source parameters from one named hyperposterior row."""
        rng = np.random.default_rng() if rng is None else rng
        if n <= 0:
            raise ValueError("n must be positive")
        self.validate_hyperparameters(hyper_row)
        if self.config.warn_if_unvalidated and not self._warned:
            warnings.warn(
                "The GWTC-4 joint sampler uses named hyperposterior rows and "
                "conditional mass sampling, but its 1D projections should be "
                "validated against rates_on_grids before publication use.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned = True

        out: dict[str, np.ndarray] = {}
        out.update(self.sample_masses(hyper_row, n, rng=rng))
        out.update(self.sample_redshift(hyper_row, n, rng=rng))
        out.update(self.sample_spins(hyper_row, n, rng=rng))
        if include_extrinsics:
            out.update(sample_isotropic_extrinsics(n, rng=rng))
        df = pd.DataFrame(out)
        df["model_name"] = self.name
        df["sampler_mode"] = "gwtc4_named_joint_model"
        df["preserves_joint_covariance"] = True
        df["joint_sampler_validation_status"] = "requires_rates_on_grids_validation"
        return df

    def sample_masses(
        self,
        row: Mapping[str, float],
        n: int,
        *,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        cfg = self.config
        mlow_1 = _required(row, "mlow_1")
        mlow_2 = _required(row, "mlow_2")
        mmax = _required(row, "mmax")

        lo = max(cfg.m1_min, min(mlow_1, mlow_2) * 0.8)
        hi = min(cfg.m1_max, mmax)
        if lo >= hi:
            raise ModelValidationError(f"Invalid m1 grid limits: lo={lo}, hi={hi}")

        m1_grid = np.linspace(lo, hi, cfg.mass_grid_size)
        p_m1 = self.primary_mass_pdf(m1_grid, row)
        m1 = InverseCDFSampler.from_pdf(m1_grid, p_m1).sample(n, rng)

        q = np.empty(n)
        q_grid_base = np.linspace(cfg.q_min, cfg.q_max, cfg.q_grid_size)
        for i, mass_1 in enumerate(m1):
            q_min = max(cfg.q_min, mlow_2 / mass_1)
            q_grid = q_grid_base[q_grid_base >= q_min]
            if len(q_grid) < 2:
                q[i] = min(1.0, max(q_min, cfg.q_min))
                continue
            p_q = self.conditional_mass_ratio_pdf(q_grid, mass_1, row)
            q[i] = InverseCDFSampler.from_pdf(q_grid, p_q).sample(1, rng)[0]

        m2 = q * m1
        chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        return {
            "mass_1_source": m1,
            "mass_2_source": m2,
            "mass_ratio": q,
            "chirp_mass_source": chirp,
            "total_mass_source": m1 + m2,
        }

    def primary_mass_pdf(self, mass_1: np.ndarray, row: Mapping[str, float]) -> np.ndarray:
        """Unnormalized p(mass_1 | Λ) for the named GWTC-4 mass model."""
        x = np.asarray(mass_1, dtype=float)
        alpha_1 = _required(row, "alpha_1")
        alpha_2 = _required(row, "alpha_2")
        break_mass = self._resolve_break_mass(
            _required(row, "break_mass"),
            _required(row, "mlow_1"),
            _required(row, "mmax"),
        )
        delta_m_1 = _required(row, "delta_m_1")
        lam_0 = np.clip(_required(row, "lam_0"), 0.0, 1.0)
        lam_1 = np.clip(_required(row, "lam_1"), 0.0, 1.0 - lam_0)
        continuum_weight = max(0.0, 1.0 - lam_0 - lam_1)

        mlow_1 = _required(row, "mlow_1")
        mmax = _required(row, "mmax")
        support = (x >= mlow_1) & (x <= mmax)
        smoothing = self._low_mass_smoothing(x, mlow_1, delta_m_1)

        continuum = np.zeros_like(x)
        below = support & (x <= break_mass)
        above = support & (x > break_mass)
        continuum[below] = np.power(np.maximum(x[below] / break_mass, 1e-300), alpha_1)
        continuum[above] = np.power(np.maximum(x[above] / break_mass, 1e-300), alpha_2)
        continuum *= smoothing

        peak_0 = self._normal_pdf(x, _required(row, "mpp_1"), _required(row, "sigpp_1")) * support * smoothing
        peak_1 = self._normal_pdf(x, _required(row, "mpp_2"), _required(row, "sigpp_2")) * support * smoothing

        continuum = _trapz_normalize(x, continuum, label="primary-mass continuum")
        peak_0 = _trapz_normalize(x, peak_0, label="primary-mass peak 0")
        peak_1 = _trapz_normalize(x, peak_1, label="primary-mass peak 1")
        return continuum_weight * continuum + lam_0 * peak_0 + lam_1 * peak_1

    def conditional_mass_ratio_pdf(
        self,
        q: np.ndarray,
        mass_1: float,
        row: Mapping[str, float],
    ) -> np.ndarray:
        """Unnormalized p(q | mass_1, Λ)."""
        q = np.asarray(q, dtype=float)
        beta = _required(row, "beta")
        mlow_2 = _required(row, "mlow_2")
        delta_m_2 = _required(row, "delta_m_2")
        mass_2 = q * float(mass_1)
        support = (q > 0) & (q <= 1.0) & (mass_2 >= 0)
        pdf = np.where(support, np.power(np.maximum(q, 1e-300), beta), 0.0)
        pdf *= self._low_mass_smoothing(mass_2, mlow_2, delta_m_2)
        return np.clip(pdf, 0.0, np.inf)

    @staticmethod
    def _resolve_break_mass(value: float, mlow: float, mmax: float) -> float:
        """Interpret break_mass as an absolute mass unless it looks fractional."""
        if 0.0 < value < 1.0:
            return mlow + value * (mmax - mlow)
        return float(np.clip(value, mlow, mmax))

    @staticmethod
    def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        sigma = max(float(sigma), 1e-12)
        return np.exp(-0.5 * ((np.asarray(x) - float(mu)) / sigma) ** 2) / sigma

    @staticmethod
    def _low_mass_smoothing(x: np.ndarray, mlow: float, delta_m: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if delta_m <= 0:
            return np.where(x >= mlow, 1.0, 0.0)
        y = (x - mlow) / delta_m
        out = np.zeros_like(y, dtype=float)
        out[y >= 1.0] = 1.0
        mask = (y > 0.0) & (y < 1.0)
        # Stable implementation of 1/(exp(1/y + 1/(y-1)) + 1).
        exponent = 1.0 / y[mask] + 1.0 / (y[mask] - 1.0)
        out[mask] = 1.0 / (np.exp(exponent) + 1.0)
        return out

    def sample_redshift(
        self,
        row: Mapping[str, float],
        n: int,
        *,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        lamb = _required(row, "lamb")
        z_grid = np.linspace(self.config.z_min, self.config.z_max, self.config.z_grid_size)
        pdf = power_law_redshift_pdf(z_grid, lamb=lamb, cosmology=self.config.cosmology)
        z = InverseCDFSampler.from_pdf(z_grid, pdf).sample(n, rng)
        return {"redshift": z, "luminosity_distance": luminosity_distance_from_redshift(z, self.config.cosmology)}

    def sample_spins(
        self,
        row: Mapping[str, float],
        n: int,
        *,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        xi = _required(row, "xi_spin")

        a1 = self._sample_truncated_normal(
            low=0.0,
            high=1.0,
            mu=_required(row, "mu_chi"),
            sigma=_required(row, "sigma_chi"),
            n=n,
            rng=rng,
        )
        a2 = self._sample_truncated_normal(
            low=0.0,
            high=1.0,
            mu=_required(row, "mu_chi"),
            sigma=_required(row, "sigma_chi"),
            n=n,
            rng=rng,
        )

        is_gaussian_binary = rng.binomial(1, xi, size=n).astype(bool)
        cos1 = np.empty(n)
        cos2 = np.empty(n)
        n_gauss = int(np.sum(is_gaussian_binary))
        n_iso = n - n_gauss

        if n_gauss > 0:
            cos1[is_gaussian_binary] = self._sample_truncated_normal(
                low=-1.0,
                high=1.0,
                mu=_required(row, "mu_spin"),
                sigma=_required(row, "sigma_spin"),
                n=n_gauss,
                rng=rng,
            )
            cos2[is_gaussian_binary] = self._sample_truncated_normal(
                low=-1.0,
                high=1.0,
                mu=_required(row, "mu_spin"),
                sigma=_required(row, "sigma_spin"),
                n=n_gauss,
                rng=rng,
            )
        if n_iso > 0:
            cos1[~is_gaussian_binary] = rng.uniform(-1.0, 1.0, size=n_iso)
            cos2[~is_gaussian_binary] = rng.uniform(-1.0, 1.0, size=n_iso)

        cos1 = np.clip(cos1, -1.0, 1.0)
        cos2 = np.clip(cos2, -1.0, 1.0)

        return {
            "a_1": a1,
            "a_2": a2,
            "cos_tilt_1": cos1,
            "cos_tilt_2": cos2,
            "tilt_1": np.arccos(cos1),
            "tilt_2": np.arccos(cos2),
            "phi_12": rng.uniform(0.0, 2.0 * np.pi, size=n),
            "phi_jl": rng.uniform(0.0, 2.0 * np.pi, size=n),
            "spin_sampler_mode": np.full(n, "joint_gaussian_component_spins", dtype=object),
            "spin_gaussian_binary_channel": is_gaussian_binary,
        }

    def _sample_truncated_normal(
        self,
        *,
        low: float,
        high: float,
        mu: float,
        sigma: float,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        grid = np.linspace(low, high, self.config.spin_grid_size)
        pdf = self._normal_pdf(grid, mu, sigma)
        pdf = np.where((grid >= low) & (grid <= high), pdf, 0.0)
        pdf = _trapz_normalize(grid, pdf, label=f"truncated normal [{low}, {high}]")
        return InverseCDFSampler.from_pdf(grid, pdf).sample(n, rng)


# Backwards-compatible public name used by the current CLI and posterior sampler.
BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift = (
    GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift
)