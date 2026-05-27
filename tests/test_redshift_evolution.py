from __future__ import annotations

import pytest

from popsampler.redshift_evolution import (
    ParameterEvolution,
    RedshiftEvolutionConfig,
    evolve_power_law,
    evolve_row_power_law,
    evolve_value,
)
from popsampler.scripts.sample_bbh import parse_evolution_spec


def test_power_law_evolution_matches_one_plus_z_scaling() -> None:
    assert evolve_power_law(40.0, 1.0, 1.0) == pytest.approx(80.0)
    assert evolve_power_law(40.0, 1.0, -1.0) == pytest.approx(20.0)


def test_linear_log1pz_evolution() -> None:
    value = evolve_value(10.0, 1.0, 2.0, kind="linear_log1pz")
    assert value == pytest.approx(10.0 + 2.0 * 0.6931471805599453)


def test_evolve_row_power_law_uses_row_specific_exponent() -> None:
    row = {"mpp_2": 35.0, "gamma_mpp_2": 1.0, "beta": -2.0}
    evolved = evolve_row_power_law(row, 1.0, {"mpp_2": "gamma_mpp_2"})
    assert evolved["mpp_2"] == pytest.approx(70.0)
    assert evolved["beta"] == pytest.approx(-2.0)


def test_redshift_evolution_config_applies_multiple_parameters() -> None:
    row = {"mpp_2": 35.0, "beta": -2.0, "xi_spin": 0.4, "gamma_mpp_2": 1.0}
    config = RedshiftEvolutionConfig(
        enabled=True,
        parameter_evolutions=(
            ParameterEvolution("mpp_2", "gamma_mpp_2", kind="power_law"),
            ParameterEvolution("beta", -0.5, kind="linear_z"),
            ParameterEvolution("xi_spin", 2.0, kind="linear_z", clip_min=0.0, clip_max=1.0),
        ),
    )
    evolved = config.evolve_row(row, 1.0)
    assert evolved["mpp_2"] == pytest.approx(70.0)
    assert evolved["beta"] == pytest.approx(-2.5)
    assert evolved["xi_spin"] == pytest.approx(1.0)
    assert "mpp_2:power_law:gamma_mpp_2" in config.describe()


def test_disabled_config_leaves_row_unchanged() -> None:
    row = {"mpp_2": 35.0}
    assert RedshiftEvolutionConfig.disabled().evolve_row(row, 1.0) == row


def test_parse_evolution_spec_accepts_numeric_and_column_coefficients() -> None:
    numeric = parse_evolution_spec("beta:linear_log1pz:-0.5")
    assert numeric.parameter == "beta"
    assert numeric.kind == "linear_log1pz"
    assert numeric.coefficient == pytest.approx(-0.5)

    column = parse_evolution_spec("mpp_2:power_law:gamma_mpp_2")
    assert column.parameter == "mpp_2"
    assert column.kind == "power_law"
    assert column.coefficient == "gamma_mpp_2"

    clipped = parse_evolution_spec("xi_spin:linear_z:gamma_xi:0:1")
    assert clipped.clip_min == pytest.approx(0.0)
    assert clipped.clip_max == pytest.approx(1.0)


def test_bad_evolution_spec_raises() -> None:
    with pytest.raises(ValueError, match="Evolution specs"):
        parse_evolution_spec("beta:-0.5")
    with pytest.raises(ValueError, match="Unknown evolution kind"):
        parse_evolution_spec("beta:quadratic:-0.5")
