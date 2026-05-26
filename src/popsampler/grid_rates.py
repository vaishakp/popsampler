"""Accessors for popsummary `rates_on_grids` products.

The GWTC-4 popsummary files provide posterior samples of one-dimensional rate
functions evaluated on fixed grids, for example p(mass_1), p(mass_ratio),
p(redshift), and spin marginals.

These grids are first-class products for inspection, plotting, and validation.
They are **not** the event sampler used for CE catalogs because independently
sampling these one-dimensional marginals would not preserve joint covariance
or conditional structure such as mass_1--mass_ratio or q--chi_eff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import h5py
import numpy as np

from .samplers import InverseCDFSampler


@dataclass(frozen=True)
class RateGrid:
    """A one-dimensional marginal rate grid for all hyperposterior rows.

    This class deliberately represents a *marginal* product. Use it to access,
    plot, and validate released one-dimensional rates. Do not use independent
    draws from several RateGrid objects as a covariance-preserving BBH event
    sampler.
    """

    name: str
    positions: np.ndarray
    rates: np.ndarray

    @property
    def n_hyperposterior_samples(self) -> int:
        return int(self.rates.shape[0])

    def rate_for_row(self, row_index: int) -> np.ndarray:
        """Return the unnormalized marginal rate curve for one hyperposterior row."""
        return np.asarray(self.rates[row_index], dtype=float)

    @property
    def mean_rate(self) -> np.ndarray:
        """Posterior mean marginal rate curve over hyperposterior rows."""
        return np.asarray(self.rates.mean(axis=0), dtype=float)

    def quantile_rate(self, quantile: float) -> np.ndarray:
        """Pointwise posterior quantile of the marginal rate curve."""
        return np.asarray(np.quantile(self.rates, quantile, axis=0), dtype=float)

    def normalized_rate(self, row_index: int | None = None) -> np.ndarray:
        """Return an area-normalized marginal PDF.

        If ``row_index`` is omitted, normalize the posterior-mean rate curve.
        """
        rate = self.mean_rate if row_index is None else self.rate_for_row(row_index)
        rate = np.clip(rate, 0.0, np.inf)
        norm = np.trapz(rate, self.positions)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Rate grid {self.name!r} has non-positive integral")
        return rate / norm

    def sample_for_row(
        self,
        row_index: int,
        n: int,
        *,
        rng: np.random.Generator | None = None,
        allow_validation_sampling: bool = False,
    ) -> np.ndarray:
        """Draw samples from one normalized 1D marginal rate grid.

        This method is retained only for validation and debugging of the released
        marginal grids. It intentionally requires ``allow_validation_sampling``
        so that production code does not accidentally use independent 1D
        marginals as the BBH event sampler.
        """
        if not allow_validation_sampling:
            raise RuntimeError(
                "RateGrid.sample_for_row samples a released 1D marginal only. "
                "It does not preserve joint covariance and must not be used as "
                "the BBH event sampler. Pass allow_validation_sampling=True only "
                "inside validation/debugging scripts."
            )
        warnings.warn(
            "Sampling from RateGrid uses a released 1D marginal and does not "
            "preserve joint covariance. This is validation-only functionality.",
            RuntimeWarning,
            stacklevel=2,
        )
        rng = np.random.default_rng() if rng is None else rng
        pdf = self.rate_for_row(row_index)
        return InverseCDFSampler.from_pdf(self.positions, pdf).sample(n, rng)


def list_rate_grid_names(path: str | Path) -> list[str]:
    """List names under `posterior/rates_on_grids`."""
    with h5py.File(path, "r") as h5:
        group = h5.get("posterior/rates_on_grids")
        if group is None:
            return []
        return sorted(group.keys())


def load_rate_grid(path: str | Path, name: str) -> RateGrid:
    """Load one one-dimensional marginal rate grid from a popsummary file."""
    with h5py.File(path, "r") as h5:
        base = f"posterior/rates_on_grids/{name}"
        if base not in h5:
            available = list_rate_grid_names(path)
            raise KeyError(f"No rate grid named {name!r}. Available grids: {available}")
        positions = np.asarray(h5[f"{base}/positions"])
        rates = np.asarray(h5[f"{base}/rates"])

    # The release stores positions as shape (1, n_grid) for 1D grids.
    positions = np.squeeze(positions)
    if positions.ndim != 1:
        raise ValueError(f"Expected a one-dimensional position grid for {name!r}; got {positions.shape}")
    if rates.ndim != 2:
        raise ValueError(f"Expected rates shape (n_hyper, n_grid) for {name!r}; got {rates.shape}")
    if rates.shape[1] != len(positions):
        raise ValueError(
            f"Grid/rate mismatch for {name!r}: positions={positions.shape}, rates={rates.shape}"
        )
    return RateGrid(name=name, positions=positions, rates=rates)
