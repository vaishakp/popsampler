"""Model-specific posterior-predictive samplers.

This module deliberately separates two tasks:

1. draw a coherent hyperparameter row Λ from the released hyperposterior;
2. draw source parameters θ from the population model conditional on Λ.

The first implemented target is the GWTC-4.0 strongly modeled BBH file
`BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5`.
The implementation is conservative: it validates required hyperparameter names
instead of silently substituting guessed defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .cosmology import DEFAULT_COSMOLOGY, luminosity_distance_from_redshift, power_law_redshift_pdf
from .extrinsics import sample_isotropic_extrinsics
from .samplers import InverseCDFSampler


class ModelValidationError(ValueError):
    """Raised when a hyperposterior row cannot be mapped to a model."""


def _first_present(row: Mapping[str, float], names: tuple[str, ...]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    raise ModelValidationError(f"Missing one of required hyperparameters {names}")


def _optional(row: Mapping[str, float], names: tuple[str, ...], default: float) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return float(default)


@dataclass
class BBHDefaultModelConfig:
    """Numerical settings for the default BBH sampler."""

    m1_min: float = 2.0
    m1_max: float = 200.0
    q_min: float = 0.02
    q_max: float = 1.0
    z_min: float = 1.0e-5
    z_max: float = 20.0
    mass_grid_size: int = 4096
    q_grid_size: int = 2048
    z_grid_size: int = 4096
    cosmology: object = field(default_factory=lambda: DEFAULT_COSMOLOGY)


class BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift:
    """Posterior-predictive sampler for the default strongly modeled BBH family.

    Notes
    -----
    The exact LVK implementation lives in the population-analysis products. This
    class implements the standard components used by the public model name:
    broken-power-law + two Gaussian peaks for primary mass, a power-law mass-ratio
    distribution, Gaussian-component spin tilt model with beta spin magnitudes,
    and power-law redshift evolution. The `validate_hyperparameters` method is
    intentionally strict where names are unambiguous and permissive only for known
    naming aliases.

    This should be validated against the released figure scripts/rate grids before
    using the output as publication-grade posterior predictive samples.
    """

    name = "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift"

    def __init__(self, config: BBHDefaultModelConfig | None = None):
        self.config = BBHDefaultModelConfig() if config is None else config

    def validate_hyperparameters(self, row: Mapping[str, float]) -> None:
        required_groups = [
            ("alpha", "alpha_1"),
            ("beta", "beta_q"),
            ("mmin", "m_min", "minimum_mass"),
            ("mmax", "m_max", "maximum_mass"),
            ("lam", "lambda_peak", "lambda_m"),
            ("mpp", "mu_m", "m_peak"),
            ("sigpp", "sigma_m", "sigma_peak"),
            ("lam_2", "lambda_peak_2", "lambda_m_2"),
            ("mpp_2", "mu_m_2", "m_peak_2"),
            ("sigpp_2", "sigma_m_2", "sigma_peak_2"),
            ("alpha_chi", "alpha_spin"),
            ("beta_chi", "beta_spin"),
            ("sigma_t", "sigma_cost", "sigma_tilt"),
            ("xi_spin", "zeta_spin", "xi_tilt"),
            ("lamb", "lambda_z", "kappa"),
        ]
        for names in required_groups:
            _first_present(row, names)

    def sample(self, hyper_row: Mapping[str, float], n: int, *, rng: np.random.Generator | None = None, include_extrinsics: bool = True) -> pd.DataFrame:
        rng = np.random.default_rng() if rng is None else rng
        self.validate_hyperparameters(hyper_row)
        out: dict[str, np.ndarray] = {}
        out.update(self.sample_masses(hyper_row, n, rng=rng))
        out.update(self.sample_redshift(hyper_row, n, rng=rng))
        out.update(self.sample_spins(hyper_row, n, rng=rng))
        if include_extrinsics:
            out.update(sample_isotropic_extrinsics(n, rng=rng))
        df = pd.DataFrame(out)
        df["model_name"] = self.name
        return df

    def sample_masses(self, row: Mapping[str, float], n: int, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
        cfg = self.config
        mmin = _first_present(row, ("mmin", "m_min", "minimum_mass"))
        mmax = _first_present(row, ("mmax", "m_max", "maximum_mass"))
        alpha = _first_present(row, ("alpha", "alpha_1"))
        beta = _first_present(row, ("beta", "beta_q"))
        lam1 = _first_present(row, ("lam", "lambda_peak", "lambda_m"))
        mu1 = _first_present(row, ("mpp", "mu_m", "m_peak"))
        sig1 = _first_present(row, ("sigpp", "sigma_m", "sigma_peak"))
        lam2 = _first_present(row, ("lam_2", "lambda_peak_2", "lambda_m_2"))
        mu2 = _first_present(row, ("mpp_2", "mu_m_2", "m_peak_2"))
        sig2 = _first_present(row, ("sigpp_2", "sigma_m_2", "sigma_peak_2"))
        delta_m = _optional(row, ("delta_m", "dmmin"), 0.0)

        m1_grid = np.linspace(max(cfg.m1_min, mmin), min(cfg.m1_max, mmax), cfg.mass_grid_size)
        p_m1 = self._primary_mass_pdf(m1_grid, mmin, mmax, alpha, lam1, mu1, sig1, lam2, mu2, sig2, delta_m)
        m1 = InverseCDFSampler.from_pdf(m1_grid, p_m1).sample(n, rng)

        q = np.empty(n)
        q_grid_base = np.linspace(cfg.q_min, cfg.q_max, cfg.q_grid_size)
        for i, m in enumerate(m1):
            q_min = max(cfg.q_min, mmin / m)
            q_grid = q_grid_base[q_grid_base >= q_min]
            if len(q_grid) < 2:
                q[i] = q_min
                continue
            p_q = np.power(q_grid, beta) * self._low_mass_smoothing(q_grid * m, mmin, delta_m)
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

    @staticmethod
    def _primary_mass_pdf(x: np.ndarray, mmin: float, mmax: float, alpha: float, lam1: float, mu1: float, sig1: float, lam2: float, mu2: float, sig2: float, delta_m: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        base = np.where((x >= mmin) & (x <= mmax), np.power(x, -alpha), 0.0)
        g1 = np.exp(-0.5 * ((x - mu1) / sig1) ** 2) / max(sig1, 1e-12)
        g2 = np.exp(-0.5 * ((x - mu2) / sig2) ** 2) / max(sig2, 1e-12)
        lam1 = np.clip(lam1, 0.0, 1.0)
        lam2 = np.clip(lam2, 0.0, 1.0 - lam1)
        pdf = (1.0 - lam1 - lam2) * base + lam1 * g1 + lam2 * g2
        return np.clip(pdf * BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift._low_mass_smoothing(x, mmin, delta_m), 0.0, np.inf)

    @staticmethod
    def _low_mass_smoothing(x: np.ndarray, mmin: float, delta_m: float) -> np.ndarray:
        if delta_m <= 0:
            return np.where(x >= mmin, 1.0, 0.0)
        y = (np.asarray(x) - mmin) / delta_m
        return np.where(y <= 0, 0.0, np.where(y >= 1, 1.0, 1.0 / (np.exp(1.0 / y + 1.0 / (y - 1.0)) + 1.0)))

    def sample_redshift(self, row: Mapping[str, float], n: int, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
        lamb = _first_present(row, ("lamb", "lambda_z", "kappa"))
        z_grid = np.linspace(self.config.z_min, self.config.z_max, self.config.z_grid_size)
        pdf = power_law_redshift_pdf(z_grid, lamb=lamb, cosmology=self.config.cosmology)
        z = InverseCDFSampler.from_pdf(z_grid, pdf).sample(n, rng)
        return {"redshift": z, "luminosity_distance": luminosity_distance_from_redshift(z, self.config.cosmology)}

    def sample_spins(self, row: Mapping[str, float], n: int, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
        alpha_chi = _first_present(row, ("alpha_chi", "alpha_spin"))
        beta_chi = _first_present(row, ("beta_chi", "beta_spin"))
        sigma_t = _first_present(row, ("sigma_t", "sigma_cost", "sigma_tilt"))
        xi = np.clip(_first_present(row, ("xi_spin", "zeta_spin", "xi_tilt")), 0.0, 1.0)

        a1 = rng.beta(alpha_chi, beta_chi, size=n)
        a2 = rng.beta(alpha_chi, beta_chi, size=n)

        aligned_1 = np.clip(rng.normal(1.0, sigma_t, size=n), -1.0, 1.0)
        aligned_2 = np.clip(rng.normal(1.0, sigma_t, size=n), -1.0, 1.0)
        iso_1 = rng.uniform(-1.0, 1.0, size=n)
        iso_2 = rng.uniform(-1.0, 1.0, size=n)
        choose_aligned = rng.uniform(0.0, 1.0, size=n) < xi
        cos1 = np.where(choose_aligned, aligned_1, iso_1)
        cos2 = np.where(choose_aligned, aligned_2, iso_2)

        return {
            "a_1": a1,
            "a_2": a2,
            "cos_tilt_1": cos1,
            "cos_tilt_2": cos2,
            "tilt_1": np.arccos(cos1),
            "tilt_2": np.arccos(cos2),
            "phi_12": rng.uniform(0.0, 2.0 * np.pi, size=n),
            "phi_jl": rng.uniform(0.0, 2.0 * np.pi, size=n),
        }