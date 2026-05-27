from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from popsampler.analytic_marginals import (
    analytic_1d_pdf_for_rows,
    compare_curves,
    cos_tilt_pdf,
    mass_ratio_joint_pdf,
    normalize_pdf,
    normalize_pdf_2d,
    row_normalized_grid_target,
    spin_magnitude_pdf,
)
from popsampler.models import BBHDefaultModelConfig, GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift


@pytest.fixture
def example_row() -> dict[str, float]:
    return {
        "alpha_1": 3.0,
        "alpha_2": 4.0,
        "beta": 1.5,
        "break_mass": 35.0,
        "delta_m_1": 4.0,
        "delta_m_2": 4.0,
        "lam_0": 0.1,
        "lam_1": 0.4,
        "mlow_1": 5.0,
        "mlow_2": 5.0,
        "mmax": 100.0,
        "mpp_1": 35.0,
        "mpp_2": 65.0,
        "sigpp_1": 3.0,
        "sigpp_2": 5.0,
        "mu_chi": 0.25,
        "sigma_chi": 0.12,
        "mu_spin": 0.6,
        "sigma_spin": 0.25,
        "xi_spin": 0.7,
        "lamb": 2.0,
    }


def test_normalize_pdf_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_pdf(np.array([0.0, 1.0, 1.0]), np.ones(3))


def test_row_normalized_grid_target_averages_pdfs() -> None:
    x = np.linspace(0.0, 1.0, 256)
    rates = np.vstack([np.ones_like(x), 2.0 * x])

    target = row_normalized_grid_target(x, rates)

    expected = normalize_pdf(x, 0.5 * normalize_pdf(x, rates[0]) + 0.5 * normalize_pdf(x, rates[1]))
    assert np.allclose(target, expected)


def test_spin_magnitude_and_tilt_pdfs_normalize(example_row: dict[str, float]) -> None:
    a = np.linspace(0.0, 1.0, 512)
    c = np.linspace(-1.0, 1.0, 512)

    p_a = spin_magnitude_pdf(example_row, a)
    p_c = cos_tilt_pdf(example_row, c)

    assert np.trapezoid(p_a, a) == pytest.approx(1.0)
    assert np.trapezoid(p_c, c) == pytest.approx(1.0)
    assert np.all(p_a >= 0.0)
    assert np.all(p_c >= 0.0)


def test_mass_ratio_joint_normalizes(example_row: dict[str, float]) -> None:
    model = GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift(
        config=BBHDefaultModelConfig(m1_max=100.0, mass_grid_size=180, q_grid_size=120, warn_if_unvalidated=False)
    )
    m1 = np.linspace(2.0, 100.0, 180)
    q = np.linspace(0.001, 1.0, 120)

    joint = mass_ratio_joint_pdf(model, example_row, m1, q)

    assert joint.shape == (len(m1), len(q))
    assert np.trapezoid(np.trapezoid(joint, q, axis=1), m1) == pytest.approx(1.0)
    assert np.all(joint >= 0.0)


def test_analytic_1d_pdf_for_rows_normalizes(example_row: dict[str, float]) -> None:
    model = GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift(
        config=BBHDefaultModelConfig(m1_max=100.0, mass_grid_size=180, q_grid_size=120, warn_if_unvalidated=False)
    )
    hyper = pd.DataFrame([example_row, {**example_row, "mu_chi": 0.35, "lamb": 1.0}])
    grids = {
        "mass_1": np.linspace(2.0, 100.0, 180),
        "mass_ratio": np.linspace(0.001, 1.0, 120),
        "a_1": np.linspace(0.0, 1.0, 128),
        "a_2": np.linspace(0.0, 1.0, 128),
        "cos_tilt_1": np.linspace(-1.0, 1.0, 128),
        "cos_tilt_2": np.linspace(-1.0, 1.0, 128),
        "redshift": np.linspace(0.0, 1.9, 128),
    }

    for name, x in grids.items():
        pdf = analytic_1d_pdf_for_rows(model, hyper, [0, 1], name, x)
        assert np.trapezoid(pdf, x) == pytest.approx(1.0)
        assert np.all(pdf >= 0.0)


def test_compare_curves_detects_identical_curves() -> None:
    x = np.linspace(0.0, 1.0, 256)
    pdf = normalize_pdf(x, 1.0 + x)

    comparison = compare_curves("toy", x, pdf, pdf)

    assert comparison.l1 == pytest.approx(0.0)
    assert comparison.mean_abs_diff == pytest.approx(0.0)


def test_normalize_pdf_2d_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="Expected density shape"):
        normalize_pdf_2d(np.arange(3), np.arange(4), np.ones((3, 3)))
