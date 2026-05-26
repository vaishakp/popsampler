from __future__ import annotations

import numpy as np
import pytest

from popsampler.samplers import InverseCDFSampler, weighted_resample


def test_inverse_cdf_sampler_rejects_nonmonotonic_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        InverseCDFSampler.from_pdf(np.array([0.0, 1.0, 1.0]), np.ones(3))


def test_inverse_cdf_sampler_rejects_zero_pdf() -> None:
    with pytest.raises(ValueError, match="non-positive integral"):
        InverseCDFSampler.from_pdf(np.linspace(0.0, 1.0, 8), np.zeros(8))


def test_inverse_cdf_sampler_uniform_mean_is_reasonable() -> None:
    rng = np.random.default_rng(1234)
    x = np.linspace(0.0, 1.0, 256)
    sampler = InverseCDFSampler.from_pdf(x, np.ones_like(x))
    draws = sampler.sample(50_000, rng)
    assert abs(float(np.mean(draws)) - 0.5) < 0.01
    assert np.all((draws >= 0.0) & (draws <= 1.0))


def test_weighted_resample_returns_reasonable_ess() -> None:
    rng = np.random.default_rng(1234)
    values = np.array([10, 20, 30])
    weights = np.array([1.0, 1.0, 2.0])
    samples, ess = weighted_resample(values, weights, 1000, rng)
    assert samples.shape == (1000,)
    assert set(np.unique(samples)).issubset(set(values))
    expected_ess = 1.0 / np.sum((weights / weights.sum()) ** 2)
    assert ess == pytest.approx(expected_ess)


def test_weighted_resample_rejects_bad_weights() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        weighted_resample(np.arange(2), np.array([1.0, -1.0]), 10)
    with pytest.raises(ValueError, match="non-positive sum"):
        weighted_resample(np.arange(2), np.array([0.0, 0.0]), 10)
