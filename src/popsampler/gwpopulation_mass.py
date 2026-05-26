"""Use gwpopulation mass-model machinery for posterior-predictive mass draws.

The public gwpopulation 1.3.1 release provides the authoritative primary-mass
component functions for LVK-style broken-power-law and Gaussian-peak models, but
it does not expose a ready-made GWTC-4 two-peak class with separate primary and
secondary low-mass smoothing scales. This module therefore uses gwpopulation for
the primary-mass components and implements the asymmetric conditional
``p(q | mass_1)`` layer needed by the GWTC-4 parameter names ``mlow_2`` and
``delta_m_2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .samplers import InverseCDFSampler


class GWPopulationMassModelError(RuntimeError):
    pass


CANDIDATE_CLASS_NAMES = (
    "GWTC4AsymmetricBrokenPowerLawTwoPeaksMassDistribution",
    "MultiPeakSmoothedMassDistribution",
    "BrokenPowerLawPeakSmoothedMassDistribution",
    "BrokenPowerLawSmoothedMassDistribution",
)


@dataclass(frozen=True)
class GWPopulationMassSamplerConfig:
    m1_min: float = 2.0
    m1_max: float = 300.0
    q_min: float = 0.001
    q_max: float = 1.0
    mass_grid_size: int = 1200
    q_grid_size: int = 500
    gaussian_mass_maximum: float = 100.0


class GWPopulationMassSampler:
    """Adapter: gwpopulation evaluates primary components; popsampler samples."""

    def __init__(self, config: GWPopulationMassSamplerConfig | None = None):
        self.config = GWPopulationMassSamplerConfig() if config is None else config
        try:
            import gwpopulation.models.mass as gwpop_mass  # type: ignore
        except Exception as exc:
            raise GWPopulationMassModelError(
                "Could not import gwpopulation.models.mass. Install with `python -m pip install -e '.[gwtc4]'`."
            ) from exc
        self.gwpop_mass = gwpop_mass
        self._model_name = CANDIDATE_CLASS_NAMES[0]

    @property
    def model_name(self) -> str:
        return self._model_name

    def sample(self, row: Mapping[str, float], n: int, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
        m1_grid = np.linspace(self.config.m1_min, self.config.m1_max, self.config.mass_grid_size)
        p_m1 = self.marginal_mass_1_pdf(row, m1_grid)
        m1 = InverseCDFSampler.from_pdf(m1_grid, p_m1).sample(n, rng)
        q = self.sample_mass_ratio_conditional(row, m1, rng=rng)
        m2 = q * m1
        chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        return {
            "mass_1_source": m1,
            "mass_2_source": m2,
            "mass_ratio": q,
            "chirp_mass_source": chirp,
            "total_mass_source": m1 + m2,
            "mass_sampler_backend": np.full(n, "gwpopulation_primary_asymmetric_q", dtype=object),
            "gwpopulation_mass_model": np.full(n, self.model_name, dtype=object),
        }

    def marginal_mass_1_pdf(self, row: Mapping[str, float], mass_1: np.ndarray) -> np.ndarray:
        return self.primary_mass_pdf(row, np.asarray(mass_1, dtype=float))

    def sample_mass_ratio_conditional(self, row: Mapping[str, float], mass_1: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        mass_1 = np.asarray(mass_1, dtype=float)
        q_grid = np.linspace(self.config.q_min, self.config.q_max, self.config.q_grid_size)
        pdf = self.conditional_mass_ratio_pdf(row, mass_1, q_grid)
        cdf = np.cumsum(pdf, axis=1)
        totals = cdf[:, -1]
        q = np.empty(len(mass_1), dtype=float)
        good = np.isfinite(totals) & (totals > 0)
        if np.any(good):
            cdf_good = cdf[good] / totals[good, None]
            u = rng.uniform(size=int(np.sum(good)))
            idx = np.array([np.searchsorted(cdf_good[i], u[i], side="left") for i in range(len(u))])
            q[good] = q_grid[np.clip(idx, 0, len(q_grid) - 1)]
        if np.any(~good):
            q[~good] = np.maximum(self.config.q_min, float(row["mlow_2"]) / mass_1[~good])
            q[~good] = np.clip(q[~good], self.config.q_min, self.config.q_max)
        return q

    def evaluate_joint(self, row: Mapping[str, float], mass_1: np.ndarray, mass_ratio: np.ndarray) -> np.ndarray:
        mass_1 = np.asarray(mass_1, dtype=float)
        mass_ratio = np.asarray(mass_ratio, dtype=float)
        p_m1 = self.primary_mass_pdf(row, mass_1)
        p_q = self.conditional_mass_ratio_pdf(row, mass_1, mass_ratio)
        return np.clip(p_m1 * p_q, 0.0, np.inf)

    def primary_mass_pdf(self, row: Mapping[str, float], mass_1: np.ndarray) -> np.ndarray:
        mass_1 = np.asarray(mass_1, dtype=float)
        kwargs = self._primary_kwargs(row)
        mmin = kwargs["mmin"]
        mmax = kwargs["mmax"]
        delta_m = float(row["delta_m_1"])
        smooth = self._low_mass_smoothing(mass_1, mmin, delta_m, high=mmax)

        continuum = self.gwpop_mass.double_power_law_primary_mass(
            mass_1,
            alpha_1=kwargs["alpha_1"],
            alpha_2=kwargs["alpha_2"],
            mmin=mmin,
            mmax=mmax,
            break_fraction=kwargs["break_fraction"],
        ) * smooth
        lower_peak = self._normal_component(
            mass_1,
            mu=kwargs["mpp_1"],
            sigma=kwargs["sigpp_1"],
            low=mmin,
            high=self.config.gaussian_mass_maximum,
        ) * smooth
        upper_peak = self._normal_component(
            mass_1,
            mu=kwargs["mpp_2"],
            sigma=kwargs["sigpp_2"],
            low=mmin,
            high=self.config.gaussian_mass_maximum,
        ) * smooth

        continuum = self._trapz_normalize(mass_1, continuum)
        lower_peak = self._trapz_normalize(mass_1, lower_peak)
        upper_peak = self._trapz_normalize(mass_1, upper_peak)
        lam = float(np.clip(row["lam_0"], 0.0, 1.0))
        lam_1 = float(np.clip(row["lam_1"], 0.0, 1.0))
        pdf = (1.0 - lam) * continuum + lam * (lam_1 * lower_peak + (1.0 - lam_1) * upper_peak)
        return self._trapz_normalize(mass_1, pdf)

    def conditional_mass_ratio_pdf(self, row: Mapping[str, float], mass_1: np.ndarray, mass_ratio: np.ndarray) -> np.ndarray:
        mass_1 = np.asarray(mass_1, dtype=float)
        mass_ratio = np.asarray(mass_ratio, dtype=float)
        if mass_ratio.ndim == 1 and mass_1.ndim == 1 and len(mass_ratio) != len(mass_1):
            q = np.repeat(mass_ratio[None, :], len(mass_1), axis=0)
            m1 = np.repeat(mass_1[:, None], len(mass_ratio), axis=1)
        else:
            q = mass_ratio
            m1 = mass_1
        beta = float(row["beta"])
        mlow_2 = float(row["mlow_2"])
        delta_m_2 = float(row["delta_m_2"])
        m2 = q * m1
        pdf = np.power(np.maximum(q, 1e-300), beta)
        pdf *= self._low_mass_smoothing(m2, mlow_2, delta_m_2, high=m1)
        pdf = np.where((q > 0.0) & (q <= 1.0) & (m2 <= m1), pdf, 0.0)
        if pdf.ndim == 2:
            norms = np.trapz(pdf, mass_ratio, axis=1)
            good = np.isfinite(norms) & (norms > 0)
            out = np.zeros_like(pdf)
            out[good] = pdf[good] / norms[good, None]
            return out
        return np.clip(pdf, 0.0, np.inf)

    @staticmethod
    def _dataset(mass_1: np.ndarray, mass_ratio: np.ndarray) -> dict[str, np.ndarray]:
        mass_1 = np.asarray(mass_1, dtype=float)
        mass_ratio = np.asarray(mass_ratio, dtype=float)
        return {"mass_1": mass_1, "mass_ratio": mass_ratio, "mass_2": mass_1 * mass_ratio}

    @staticmethod
    def _primary_kwargs(row: Mapping[str, float]) -> dict[str, float]:
        mmin = float(row["mlow_1"])
        mmax = float(row["mmax"])
        break_mass = float(row["break_mass"])
        if 0.0 < break_mass < 1.0:
            break_fraction = break_mass
        else:
            break_fraction = (break_mass - mmin) / (mmax - mmin)
        return {
            "alpha_1": float(row["alpha_1"]),
            "alpha_2": float(row["alpha_2"]),
            "mmin": mmin,
            "mmax": mmax,
            "break_fraction": float(np.clip(break_fraction, 0.0, 1.0)),
            "mpp_1": float(row["mpp_1"]),
            "mpp_2": float(row["mpp_2"]),
            "sigpp_1": float(row["sigpp_1"]),
            "sigpp_2": float(row["sigpp_2"]),
        }

    @staticmethod
    def _normal_component(x: np.ndarray, *, mu: float, sigma: float, low: float, high: float) -> np.ndarray:
        sigma = max(float(sigma), 1e-12)
        y = np.exp(-0.5 * ((np.asarray(x) - float(mu)) / sigma) ** 2) / sigma
        return np.where((x >= low) & (x <= high), y, 0.0)

    @staticmethod
    def _low_mass_smoothing(x: np.ndarray, low: float, delta_m: float, high: float | np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        high = np.asarray(high, dtype=float)
        if delta_m <= 0:
            return np.where((x >= low) & (x <= high), 1.0, 0.0)
        y = (x - low) / delta_m
        out = np.zeros_like(x, dtype=float)
        out[y >= 1.0] = 1.0
        mask = (y > 0.0) & (y < 1.0)
        y_clip = np.clip(y[mask], 1e-12, 1.0 - 1e-12)
        exponent = 1.0 / y_clip + 1.0 / (y_clip - 1.0)
        out[mask] = 1.0 / (np.exp(exponent) + 1.0)
        return np.where(x <= high, out, 0.0)

    @staticmethod
    def _trapz_normalize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
        norm = np.trapz(y, x)
        if not np.isfinite(norm) or norm <= 0:
            raise GWPopulationMassModelError(f"Cannot normalize mass PDF: integral={norm}")
        return y / norm
