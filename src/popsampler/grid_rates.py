"""Accessors for popsummary `rates_on_grids` products.

The GWTC-4 popsummary files provide posterior samples of one-dimensional rate
functions evaluated on fixed grids, for example p(mass_1), p(mass_ratio),
p(redshift), and spin marginals. These are useful for validating conditional
samplers and, when only marginals are needed, for drawing posterior-predictive
samples directly from the released rate grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .samplers import InverseCDFSampler


@dataclass(frozen=True)
class RateGrid:
    """A one-dimensional rate grid for all hyperposterior rows."""

    name: str
    positions: np.ndarray
    rates: np.ndarray

    @property
    def n_hyperposterior_samples(self) -> int:
        return int(self.rates.shape[0])

    def rate_for_row(self, row_index: int) -> np.ndarray:
        return np.asarray(self.rates[row_index], dtype=float)

    def sample_for_row(
        self,
        row_index: int,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw samples from the normalized rate grid for one hyperposterior row."""
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
    """Load one rate grid from a popsummary file."""
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
