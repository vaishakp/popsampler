"""Model-specific posterior-predictive BBH samplers.

This module deliberately separates two tasks:

1. draw a coherent hyperparameter row Λ from the released hyperposterior;
2. draw source parameters θ from the population model conditional on Λ.

For the GWTC-4 strongly modeled BBH file, popsampler delegates the mass PDF
conventions to ``gwpopulation`` and adds the inverse-CDF / catalog-sampling layer
on top. This keeps the authoritative mass-model convention out of local hand-
written approximations while preserving hyperposterior-row bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import warnings

import numpy as np
import pandas as pd

from .cosmology import DEFAULT_COSMOLOGY, luminosity_distance_from_redshift, power_law_redshift_pdf
from .extrinsics import sample_isotropic_extrinsics
from .gwpopulation_mass import GWPopulationMassSampler, GWPopulationMassSamplerConfig
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


def _trapz_normalize(x: np.ndarray, y: np.ndarray, *, label: str) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
    norm = np.trapz(y, x)
    if not np.isfinite(norm) or norm <= 0:
        raise ModelValidationError(f"Could not normalize {label}: integral={norm}")
    return y / norm


@dataclass
class BBHDefaultModelConfig:
    """Numerical settings for the GWTC-4 default BBH sampler."""

    m1_min: float = 2.0
    m1_max: float = 300.0
    q_min: float = 0.001
    q_max: float = 1.0
    z_min: float = 1.0e-6
    z_max: float = 20.0
    mass_grid_size: int = 1200
    q_grid_size: int = 500
    z_grid_size: int = 4096
    spin_grid_size: int = 2048
    cosmology: object = field(default_factory=lambda: DEFAULT_COSMOLOGY)
    warn_if_unvalidated: bool = True


class GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift:
    """Joint sampler for the GWTC-4 named default BBH model.

    Masses are sampled with a ``gwpopulation``-backed adapter. Spins follow the
    Gaussian-component convention visible in the release-style simulation code:
    spin magnitudes are truncated Gaussians on ``[0, 1]`` with ``mu_chi`` and
    ``sigma_chi``; each binary draws a single formation channel with probability
    ``xi_spin``. In the Gaussian channel both cos-tilts are truncated Gaussians on
    ``[-1, 1]`` with ``mu_spin`` and ``sigma_spin``; otherwise they are isotropic.
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
    SPIN_PARAMETERS = ("mu_chi", "sigma_chi", "mu_spin", "sigma_spin", "xi_spin")
    REDSHIFT_PARAMETERS = ("lamb",)
    REQUIRED_PARAMETERS = MASS_PARAMETERS + SPIN_PARAMETERS + REDSHIFT_PARAMETERS

    def __init__(self, config: BBHDefaultModelConfig | None = None):
        self.config = BBHDefaultModelConfig() if config is None else config
        self.mass_sampler = GWPopulationMassSampler(
            GWPopulationMassSamplerConfig(
                m1_min=self.config.m1_min,
                m1_max=self.config.m1_max,
                q_min=self.config.q_min,
                q_max=self.config.q_max,
                mass_grid_size=self.config.mass_grid_size,
                q_grid_size=self.config.q_grid_size,
            )
        )
        self._warned = False

    def validate_hyperparameters(self, row: Mapping[str, float]) -> None:
        missing = [name for name in self.REQUIRED_PARAMETERS if name not in row or pd.isna(row[name])]
        if missing:
            raise ModelValidationError(f"Missing required hyperparameters: {missing}")
        for name in self.REQUIRED_PARAMETERS:
            _required(row, name)

        if not (0 < _required(row, "mlow_1") < _required(row, "mmax")):
            raise ModelValidationError("Expected 0 < mlow_1 < mmax")
        if not (0 < _required(row, "mlow_2") < _required(row, "mmax")):
            raise ModelValidationError("Expected 0 < mlow_2 < mmax")
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
                "The GWTC-4 sampler uses gwpopulation-backed mass PDFs and named "
                "hyperposterior rows, but projections should still be validated "
                "against rates_on_grids before publication use.",
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
        """Sample masses via gwpopulation, not a local hand-coded convention."""
        return self.mass_sampler.sample(row, n, rng=rng)

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

    @staticmethod
    def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        sigma = max(float(sigma), 1e-12)
        return np.exp(-0.5 * ((np.asarray(x) - float(mu)) / sigma) ** 2) / sigma


# Backwards-compatible public name used by the current CLI and posterior sampler.
BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift = (
    GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift
)
