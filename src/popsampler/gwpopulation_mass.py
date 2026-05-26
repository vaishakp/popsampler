"""Use gwpopulation mass PDFs for posterior-predictive mass draws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import inspect

import numpy as np

from .samplers import InverseCDFSampler


class GWPopulationMassModelError(RuntimeError):
    pass


CANDIDATE_CLASS_NAMES = (
    "MultiPeakSmoothedMassDistribution",
    "BrokenPowerLawTwoPeaksSmoothedMassDistribution",
    "BrokenPowerLawTwoPeakSmoothedMassDistribution",
    "BrokenPowerLawPeakSmoothedMassDistribution",
    "TwoPeakSmoothedMassDistribution",
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
        dataset = self._dataset(mass_1, mass_ratio)
        values = self._call_model(model, dataset, self._row_to_kwargs(row))
        return np.clip(np.asarray(values, dtype=float).reshape(np.asarray(mass_1).shape), 0.0, np.inf)

    def _ensure_model(self, row: Mapping[str, float]) -> Any:
        if self._model is not None:
            return self._model
        errors: list[str] = []
        for name in CANDIDATE_CLASS_NAMES:
            cls = getattr(self.gwpop_mass, name, None)
            if cls is None:
                continue
            for kwargs in ({}, {"mmin": self.config.m1_min, "mmax": self.config.m1_max}):
                try:
                    model = cls(**kwargs)
                    test = self._call_model(
                        model,
                        self._dataset(np.array([10.0, 20.0]), np.array([0.8, 0.5])),
                        self._row_to_kwargs(row),
                    )
                    if np.all(np.isfinite(test)) and np.any(np.asarray(test) > 0):
                        self._model = model
                        self._model_name = name
                        return model
                    errors.append(f"{name}{kwargs}: non-positive test values {test}")
                except Exception as exc:
                    errors.append(f"{name}{kwargs}: {type(exc).__name__}: {exc}")
        available = [name for name in CANDIDATE_CLASS_NAMES if hasattr(self.gwpop_mass, name)]
        raise GWPopulationMassModelError(
            "Could not find a usable gwpopulation mass model. "
            f"Available candidates: {available}. Attempts:\n" + "\n".join(errors[:30])
        )

    @staticmethod
    def _dataset(mass_1: np.ndarray, mass_ratio: np.ndarray) -> dict[str, np.ndarray]:
        mass_1 = np.asarray(mass_1, dtype=float)
        mass_ratio = np.asarray(mass_ratio, dtype=float)
        return {"mass_1": mass_1, "mass_ratio": mass_ratio, "mass_2": mass_1 * mass_ratio}

    @staticmethod
    def _row_to_kwargs(row: Mapping[str, float]) -> dict[str, float]:
        out = {str(k): float(v) for k, v in row.items() if np.isscalar(v) and np.isfinite(float(v))}
        if "mlow_1" in out:
            out.setdefault("mmin", out["mlow_1"])
            out.setdefault("mmin_1", out["mlow_1"])
        if "mlow_2" in out:
            out.setdefault("mmin_2", out["mlow_2"])
        if "delta_m_1" in out:
            out.setdefault("delta_m", out["delta_m_1"])
        return out

    def _call_model(self, model: Any, dataset: dict[str, np.ndarray], kwargs: dict[str, float]) -> np.ndarray:
        # Some gwpopulation callables expect hyperparameters as keyword arguments;
        # some helper functions/classes read them directly from the data dict.
        dataset_with_params = dict(dataset)
        dataset_with_params.update(kwargs)
        filtered_kwargs = self._filter_kwargs(model, kwargs)
        attempts = (
            lambda: model(dataset, **kwargs),
            lambda: model(dataset_with_params),
            lambda: model(dataset_with_params, **kwargs),
            lambda: model(dataset, **filtered_kwargs),
            lambda: model(dataset_with_params, **self._filter_kwargs(model, kwargs)),
            lambda: model(dataset, kwargs),
        )
        last_exc: Exception | None = None
        for attempt in attempts:
            try:
                values = attempt()
                if isinstance(values, Mapping):
                    for key in ("mass", "pdf", "probability", "p_m1_q"):
                        if key in values:
                            return np.asarray(values[key], dtype=float)
                    raise GWPopulationMassModelError(f"gwpopulation returned mapping keys {list(values)}")
                return np.asarray(values, dtype=float)
            except Exception as exc:
                last_exc = exc
        raise GWPopulationMassModelError(f"gwpopulation call failed: {last_exc}")

    @staticmethod
    def _filter_kwargs(func: Any, kwargs: dict[str, float]) -> dict[str, float]:
        try:
            signature = inspect.signature(func)
        except Exception:
            return kwargs
        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in params}
