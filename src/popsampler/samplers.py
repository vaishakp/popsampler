"""Low-level numerical sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid


@dataclass(frozen=True)
class InverseCDFSampler:
    """One-dimensional inverse-CDF sampler on a fixed grid."""

    x: np.ndarray
    cdf: np.ndarray

    @classmethod
    def from_pdf(cls, x: np.ndarray, pdf: np.ndarray) -> "InverseCDFSampler":
        x = np.asarray(x, dtype=float)
        pdf = np.asarray(pdf, dtype=float)
        if x.ndim != 1 or pdf.ndim != 1 or len(x) != len(pdf):
            raise ValueError("x and pdf must be one-dimensional arrays with matching length")
        if len(x) < 2:
            raise ValueError("at least two grid points are required")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x grid must be strictly increasing")
        if not np.all(np.isfinite(pdf)):
            raise ValueError("pdf contains non-finite values")
        pdf = np.clip(pdf, 0.0, np.inf)
        cdf = cumulative_trapezoid(pdf, x, initial=0.0)
        norm = cdf[-1]
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("pdf has non-positive integral")
        cdf = cdf / norm
        cdf[0] = 0.0
        cdf[-1] = 1.0
        return cls(x=x, cdf=cdf)

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        u = rng.uniform(0.0, 1.0, size=size)
        return np.interp(u, self.cdf, self.x)


def weighted_resample(values: np.ndarray, weights: np.ndarray, size: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Resample rows according to weights and return effective sample size."""
    rng = np.random.default_rng() if rng is None else rng
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    if len(values) != len(weights):
        raise ValueError("values and weights must have matching length")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights have non-positive sum")
    weights = weights / total
    ess = 1.0 / np.sum(weights**2)
    idx = rng.choice(len(weights), size=size, replace=True, p=weights)
    return values[idx], float(ess)
