from __future__ import annotations

import numpy as np
import pytest

from popsampler.redshift_models import (
    RedshiftRateConfig,
    madau_dickinson_rate,
    normalize_redshift_model_name,
    redshift_pdf,
)


def test_normalize_madau_dickinson_aliases():
    assert normalize_redshift_model_name("madau_dickinson") == "madau_dickinson"
    assert normalize_redshift_model_name("madau-dickenson") == "madau_dickinson"
    assert normalize_redshift_model_name("md") == "madau_dickinson"


def test_madau_dickinson_rate_is_positive_and_turns_over():
    z = np.linspace(0.0, 10.0, 200)
    rate = madau_dickinson_rate(z, alpha=2.7, beta=2.9, z_peak=1.9)
    assert np.all(np.isfinite(rate))
    assert np.all(rate > 0.0)
    assert rate[10] > rate[0]
    assert rate[-1] < rate[np.argmax(rate)]


def test_redshift_pdf_power_law_uses_lamb_column():
    z = np.linspace(1.0e-4, 2.0, 64)
    pdf = redshift_pdf(z, {"lamb": 2.0}, RedshiftRateConfig(model="power_law"))
    assert np.all(np.isfinite(pdf))
    assert np.trapz(pdf, z) > 0.0


def test_redshift_pdf_madau_with_fixed_ini_values():
    z = np.linspace(1.0e-4, 5.0, 128)
    config = RedshiftRateConfig(
        model="madau_dickinson",
        madau_alpha=2.7,
        madau_beta=2.9,
        madau_z_peak=1.9,
    )
    pdf = redshift_pdf(z, {}, config)
    assert np.all(np.isfinite(pdf))
    assert np.trapz(pdf, z) > 0.0


def test_redshift_pdf_madau_with_hyperposterior_columns():
    z = np.linspace(1.0e-4, 5.0, 128)
    row = {"gamma": 2.7, "kappa": 2.9, "z_peak": 1.9}
    config = RedshiftRateConfig(model="madau_dickinson")
    pdf = redshift_pdf(z, row, config)
    assert np.trapz(pdf, z) > 0.0


def test_madau_requires_positive_z_peak():
    with pytest.raises(ValueError):
        madau_dickinson_rate(np.asarray([0.1]), alpha=2.7, beta=2.9, z_peak=0.0)
