"""Helpers for optional parameter evolution with redshift.

These utilities implement a deliberately simple extension layer for exploratory
models in which selected hyperparameters evolve as powers of ``1 + z``:

    theta(z) = theta_ref * [ (1 + z) / (1 + z_ref) ] ** gamma.

The GWTC-4 default strongly modeled BBH population does not require these
helpers; they are opt-in scaffolding for testing explicit mass/spin-redshift
evolution models while leaving the default sampler unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np

PowerSpec: TypeAlias = str | float | int


def resolve_power(row: Mapping[str, float], spec: PowerSpec) -> float:
    """Resolve an evolution exponent from either a number or a row key."""
    if isinstance(spec, str):
        if spec not in row:
            raise ValueError(f"Missing redshift-evolution exponent {spec!r}")
        value = float(row[spec])
    else:
        value = float(spec)
    if not np.isfinite(value):
        raise ValueError(f"Redshift-evolution exponent is not finite: {value}")
    return value


def evolve_power_law(value: float, z: float, power: float, *, reference_redshift: float = 0.0) -> float:
    """Evolve a positive scalar as a power law in ``1 + z``.

    Parameters
    ----------
    value:
        Positive reference value at ``reference_redshift``.
    z:
        Redshift where the evolved value should be evaluated.
    power:
        Exponent ``gamma`` in ``value(z) = value_ref * scale**gamma``.
    reference_redshift:
        Reference redshift ``z_ref``. Default is 0.
    """
    value = float(value)
    z = float(z)
    reference_redshift = float(reference_redshift)
    power = float(power)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Power-law redshift evolution requires a positive finite value; got {value}")
    if not np.isfinite(z) or z <= -1.0:
        raise ValueError(f"Expected finite z > -1; got {z}")
    if not np.isfinite(reference_redshift) or reference_redshift <= -1.0:
        raise ValueError(f"Expected finite reference_redshift > -1; got {reference_redshift}")
    if not np.isfinite(power):
        raise ValueError(f"Expected finite power; got {power}")
    scale = (1.0 + z) / (1.0 + reference_redshift)
    evolved = value * scale**power
    if not np.isfinite(evolved):
        raise ValueError(f"Evolved value is not finite: {evolved}")
    return float(evolved)


def evolve_row_power_law(
    row: Mapping[str, float],
    z: float,
    parameter_powers: Mapping[str, PowerSpec],
    *,
    reference_redshift: float = 0.0,
) -> dict[str, float]:
    """Return a copy of ``row`` with selected parameters evolved to redshift ``z``.

    ``parameter_powers`` maps a parameter name to either a fixed exponent or to
    the name of another row entry containing the exponent. For example::

        {"mpp_2": "gamma_mpp_2", "sigpp_2": 0.5}

    evolves ``mpp_2`` with a row-specific exponent and ``sigpp_2`` with a fixed
    exponent of 0.5.
    """
    out = dict(row)
    for parameter, power_spec in parameter_powers.items():
        if parameter not in row:
            raise ValueError(f"Cannot evolve missing parameter {parameter!r}")
        power = resolve_power(row, power_spec)
        out[parameter] = evolve_power_law(
            float(row[parameter]),
            z,
            power,
            reference_redshift=reference_redshift,
        )
    return out
