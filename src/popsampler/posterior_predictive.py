"""Posterior-predictive catalog construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift


@dataclass
class PosteriorPredictiveSampler:
    """Draw posterior-predictive source samples from hyperposterior rows."""

    hyperposterior: pd.DataFrame
    model: BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift

    def sample(
        self,
        n_events: int,
        *,
        rng: np.random.Generator | None = None,
        batch_size: int = 1,
        include_extrinsics: bool = True,
    ) -> pd.DataFrame:
        """Draw a posterior-predictive catalog.

        Each source event gets a hyperparameter row sampled with replacement from
        the released hyperposterior. `batch_size` can be increased to amortize
        expensive per-row grid construction, but `batch_size=1` is the most direct
        posterior-predictive construction.
        """
        rng = np.random.default_rng() if rng is None else rng
        if n_events <= 0:
            raise ValueError("n_events must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.hyperposterior.empty:
            raise ValueError("hyperposterior table is empty")

        pieces: list[pd.DataFrame] = []
        remaining = int(n_events)
        while remaining > 0:
            n = min(batch_size, remaining)
            row_idx = int(rng.integers(0, len(self.hyperposterior)))
            row = self.hyperposterior.iloc[row_idx].to_dict()
            df = self.model.sample(row, n, rng=rng, include_extrinsics=include_extrinsics)
            df["hyper_sample_id"] = row_idx
            pieces.append(df)
            remaining -= n
        return pd.concat(pieces, ignore_index=True)
