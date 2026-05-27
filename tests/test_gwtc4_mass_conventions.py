from __future__ import annotations

import pytest

from popsampler.gwpopulation_mass import GWPopulationMassSampler


def test_two_peak_mixture_weights_follow_gwtc4_convention() -> None:
    row = {"lam_0": 0.25, "lam_1": 0.4}

    weights = GWPopulationMassSampler._paper_mixture_weights(row)

    assert weights == pytest.approx((0.75, 0.10, 0.15))


def test_two_peak_mixture_weights_clip_to_physical_support() -> None:
    row = {"lam_0": 1.2, "lam_1": -0.2}

    weights = GWPopulationMassSampler._paper_mixture_weights(row)

    assert weights == pytest.approx((0.0, 0.0, 1.0))
