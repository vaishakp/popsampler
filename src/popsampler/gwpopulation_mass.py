"""Use gwpopulation mass-model machinery for posterior-predictive mass draws.

The public gwpopulation 1.3.1 release contains the smoothing/normalization base
class used by LVK-style mass models, but it does not expose a ready-made
`BrokenPowerLawTwoPeaksSmoothedMassDistribution` class. This module therefore
builds that missing model by subclassing gwpopulation's base smoothed mass class,
while delegating the broken-power-law and Gaussian-peak component conventions to
gwpopulation functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .samplers import InverseCDFSampler


class GWPopulationMassModelError(RuntimeError):
    pass


CANDIDATE_CLASS_NAMES = (
    "GWTC4BrokenPowerLawTwoPeaksSmoothedMassDistribution",
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


class GWPopulationMassSampler:
    """Adapter: gwpopulation evaluates the mass PDF, popsampler draws samples."""

    def __init__(self, config: GWPopulationMassSamplerConfig | None = None):
        self.config = GWPopulationMassSamplerConfig() if config is None else config
        try:
            import gwpopulation.models.mass as gwpop_mass  # type: ignore
        except Exception as exc:
            raise GWPopulationMassModelError(
                "Could not import gwpopulation.models.mass. Install with `python -m pip install -e '.[gwtc4]'`."
            ) from exc
        self.gwpop_mass = gwpop_mass
        self._model: Any | None = None
        self._model_name: str | None = None

    @property
    def model_name(self) -> str:
        return self._model_name or "undiscovered_gwpopulation_mass_model"

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
            "mass_sampler_backend": np.full(n, "gwpopulation", dtype=object),
            "gwpopulation_mass_model": np.full(n, self.model_name, dtype=object),
        }

    def marginal_mass_1_pdf(self, row: Mapping[str, float], mass_1: np.ndarray) -> np.ndarray:
        q_grid = np.linspace(self.config.q_min, self.config.q_max, self.config.q_grid_size)
        m1_mesh, q_mesh = np.meshgrid(np.asarray(mass_1, dtype=float), q_grid, indexing="ij")
        joint = self.evaluate_joint(row, m1_mesh.ravel(), q_mesh.ravel()).reshape(m1_mesh.shape)
        return np.trapz(np.clip(joint, 0.0, np.inf), q_grid, axis=1)

    def sample_mass_ratio_conditional(self, row: Mapping[str, float], mass_1: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        mass_1 = np.asarray(mass_1, dtype=float)
        q_grid = np.linspace(self.config.q_min, self.config.q_max, self.config.q_grid_size)
        m1_mesh = np.repeat(mass_1[:, None], len(q_grid), axis=1)
        q_mesh = np.repeat(q_grid[None, :], len(mass_1), axis=0)
        joint = self.evaluate_joint(row, m1_mesh.ravel(), q_mesh.ravel()).reshape(m1_mesh.shape)
        joint = np.clip(joint, 0.0, np.inf)
        cdf = np.cumsum(joint, axis=1)
        totals = cdf[:, -1]
        q = np.empty(len(mass_1), dtype=float)
        good = np.isfinite(totals) & (totals > 0)
        if np.any(good):
            cdf_good = cdf[good] / totals[good, None]
            u = rng.uniform(size=int(np.sum(good)))
            idx = np.array([np.searchsorted(cdf_good[i], u[i], side="left") for i in range(len(u))])
            q[good] = q_grid[np.clip(idx, 0, len(q_grid) - 1)]
        if np.any(~good):
            q[~good] = self.config.q_min
        return q

    def evaluate_joint(self, row: Mapping[str, float], mass_1: np.ndarray, mass_ratio: np.ndarray) -> np.ndarray:
        model = self._ensure_model(row)
        values = model(self._dataset(mass_1, mass_ratio), **self._row_to_kwargs(row))
        return np.clip(np.asarray(values, dtype=float).reshape(np.asarray(mass_1).shape), 0.0, np.inf)

    def _ensure_model(self, row: Mapping[str, float]) -> Any:
        row_mmax = max(float(row.get("mmax", self.config.m1_max)), self.config.m1_max)
        if self._model is not None:
            if getattr(self._model, "mmax", row_mmax) >= row_mmax:
                return self._model
            self._model = None
            self._model_name = None

        cls = self._build_broken_power_law_two_peak_class()
        try:
            self._model = cls(
                mmin=self.config.m1_min,
                mmax=row_mmax,
                normalization_shape=(self.config.mass_grid_size, self.config.q_grid_size),
            )
            self._model_name = cls.__name__
            test = self._model(
                self._dataset(np.array([20.0, 40.0]), np.array([0.8, 0.5])),
                **self._row_to_kwargs(row),
            )
            if not (np.all(np.isfinite(test)) and np.any(np.asarray(test) > 0)):
                raise GWPopulationMassModelError(f"non-positive/non-finite smoke-test values {test}")
            return self._model
        except Exception as exc:
            raise GWPopulationMassModelError(
                "Could not instantiate/evaluate the gwpopulation-backed GWTC-4 broken-power-law two-peak model: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _build_broken_power_law_two_peak_class(self):
        mass = self.gwpop_mass

        def broken_power_law_two_peak_primary_mass(
            mass_array,
            alpha_1,
            alpha_2,
            mmin,
            mmax,
            break_fraction,
            lam,
            lam_1,
            mpp_1,
            mpp_2,
            sigpp_1,
            sigpp_2,
            gaussian_mass_maximum=100,
        ):
            continuum = mass.double_power_law_primary_mass(
                mass_array,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                mmin=mmin,
                mmax=mmax,
                break_fraction=break_fraction,
            )
            lower_peak = mass.double_power_law_peak_primary_mass(
                mass_array,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                mmin=mmin,
                mmax=mmax,
                break_fraction=break_fraction,
                lam=1.0,
                mpp=mpp_1,
                sigpp=sigpp_1,
                gaussian_mass_maximum=gaussian_mass_maximum,
            )
            upper_peak = mass.double_power_law_peak_primary_mass(
                mass_array,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                mmin=mmin,
                mmax=mmax,
                break_fraction=break_fraction,
                lam=1.0,
                mpp=mpp_2,
                sigpp=sigpp_2,
                gaussian_mass_maximum=gaussian_mass_maximum,
            )
            lam = float(np.clip(lam, 0.0, 1.0))
            lam_1 = float(np.clip(lam_1, 0.0, 1.0))
            return (1.0 - lam) * continuum + lam * (lam_1 * lower_peak + (1.0 - lam_1) * upper_peak)

        class GWTC4BrokenPowerLawTwoPeaksSmoothedMassDistribution(mass.BaseSmoothedMassDistribution):
            primary_model = staticmethod(broken_power_law_two_peak_primary_mass)

            @property
            def kwargs(self):
                return dict(gaussian_mass_maximum=self.mmax)

        return GWTC4BrokenPowerLawTwoPeaksSmoothedMassDistribution

    @staticmethod
    def _dataset(mass_1: np.ndarray, mass_ratio: np.ndarray) -> dict[str, np.ndarray]:
        mass_1 = np.asarray(mass_1, dtype=float)
        mass_ratio = np.asarray(mass_ratio, dtype=float)
        return {"mass_1": mass_1, "mass_ratio": mass_ratio, "mass_2": mass_1 * mass_ratio}

    @staticmethod
    def _row_to_kwargs(row: Mapping[str, float]) -> dict[str, float]:
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
            "beta": float(row["beta"]),
            "mmin": mmin,
            "mmax": mmax,
            "break_fraction": float(np.clip(break_fraction, 0.0, 1.0)),
            "lam": float(row["lam_0"]),
            "lam_1": float(row["lam_1"]),
            "mpp_1": float(row["mpp_1"]),
            "mpp_2": float(row["mpp_2"]),
            "sigpp_1": float(row["sigpp_1"]),
            "sigpp_2": float(row["sigpp_2"]),
            "delta_m": float(row["delta_m_1"]),
        }
