"""Helpers for optional parameter evolution with redshift.

The default GWTC-4 strongly-modeled BBH sampler is block factorized as
``p(mass, spin, z | Lambda) = p(mass | Lambda) p(spin | Lambda) p(z | Lambda)``.
This module provides an opt-in layer for exploratory models in which selected
hyperparameters are first evolved to the sampled event redshift and the event
intrinsics are then drawn conditional on those evolved values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np

PowerSpec: TypeAlias = str | float | int
EvolutionKind: TypeAlias = Literal["power_law", "linear_z", "linear_log1pz"]


def resolve_power(row: Mapping[str, float], spec: PowerSpec) -> float:
    """Resolve an evolution coefficient from either a number or a row key."""
    if isinstance(spec, str):
        if spec not in row:
            raise ValueError(f"Missing redshift-evolution coefficient {spec!r}")
        value = float(row[spec])
    else:
        value = float(spec)
    if not np.isfinite(value):
        raise ValueError(f"Redshift-evolution coefficient is not finite: {value}")
    return value


def _validate_redshift(z: float, *, label: str) -> float:
    value = float(z)
    if not np.isfinite(value) or value <= -1.0:
        raise ValueError(f"Expected finite {label} > -1; got {value}")
    return value


def redshift_scale(z: float, *, reference_redshift: float = 0.0) -> float:
    """Return ``(1 + z) / (1 + z_ref)`` after validation."""
    z = _validate_redshift(z, label="z")
    reference_redshift = _validate_redshift(reference_redshift, label="reference_redshift")
    return float((1.0 + z) / (1.0 + reference_redshift))


def evolve_value(
    value: float,
    z: float,
    coefficient: float,
    *,
    kind: EvolutionKind = "power_law",
    reference_redshift: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> float:
    """Evolve one scalar hyperparameter to redshift ``z``.

    Supported conventions are:

    ``power_law``
        ``theta(z) = theta_ref * [(1 + z) / (1 + z_ref)] ** coefficient``.
    ``linear_z``
        ``theta(z) = theta_ref + coefficient * (z - z_ref)``.
    ``linear_log1pz``
        ``theta(z) = theta_ref + coefficient * log[(1 + z) / (1 + z_ref)]``.
    """
    value = float(value)
    coefficient = float(coefficient)
    if not np.isfinite(value):
        raise ValueError(f"Cannot evolve non-finite value: {value}")
    if not np.isfinite(coefficient):
        raise ValueError(f"Cannot use non-finite evolution coefficient: {coefficient}")
    scale = redshift_scale(z, reference_redshift=reference_redshift)

    if kind == "power_law":
        evolved = value * scale**coefficient
    elif kind == "linear_z":
        evolved = value + coefficient * (float(z) - float(reference_redshift))
    elif kind == "linear_log1pz":
        evolved = value + coefficient * np.log(scale)
    else:
        raise ValueError(f"Unknown redshift-evolution kind {kind!r}")

    if clip_min is not None:
        evolved = max(float(clip_min), float(evolved))
    if clip_max is not None:
        evolved = min(float(clip_max), float(evolved))
    if not np.isfinite(evolved):
        raise ValueError(f"Evolved value is not finite: {evolved}")
    return float(evolved)


@dataclass(frozen=True)
class ParameterEvolution:
    """Specification for how one hyperparameter evolves with redshift."""

    parameter: str
    coefficient: PowerSpec
    kind: EvolutionKind = "power_law"
    clip_min: float | None = None
    clip_max: float | None = None

    def validate(self, row: Mapping[str, float]) -> None:
        if self.parameter not in row:
            raise ValueError(f"Cannot evolve missing parameter {self.parameter!r}")
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise ValueError(
                f"Invalid clip bounds for {self.parameter!r}: {self.clip_min} > {self.clip_max}"
            )
        resolve_power(row, self.coefficient)

    def apply(self, row: Mapping[str, float], z: float, *, reference_redshift: float = 0.0) -> float:
        self.validate(row)
        return evolve_value(
            float(row[self.parameter]),
            z,
            resolve_power(row, self.coefficient),
            kind=self.kind,
            reference_redshift=reference_redshift,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )

    def describe(self) -> str:
        coeff = self.coefficient if isinstance(self.coefficient, str) else f"{float(self.coefficient):g}"
        clip = ""
        if self.clip_min is not None or self.clip_max is not None:
            clip = f",clip=[{self.clip_min},{self.clip_max}]"
        return f"{self.parameter}:{self.kind}:{coeff}{clip}"


@dataclass(frozen=True)
class RedshiftEvolutionConfig:
    """Opt-in redshift-evolution configuration for a BBH population sampler."""

    enabled: bool = False
    parameter_evolutions: tuple[ParameterEvolution, ...] = field(default_factory=tuple)
    reference_redshift: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_evolutions", tuple(self.parameter_evolutions))
        _validate_redshift(self.reference_redshift, label="reference_redshift")

    @classmethod
    def disabled(cls) -> "RedshiftEvolutionConfig":
        return cls(enabled=False)

    @classmethod
    def from_parameter_powers(
        cls,
        parameter_powers: Mapping[str, PowerSpec],
        *,
        reference_redshift: float = 0.0,
        enabled: bool = True,
    ) -> "RedshiftEvolutionConfig":
        return cls(
            enabled=enabled,
            parameter_evolutions=tuple(
                ParameterEvolution(parameter=name, coefficient=power, kind="power_law")
                for name, power in parameter_powers.items()
            ),
            reference_redshift=reference_redshift,
        )

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.parameter_evolutions)

    def validate(self, row: Mapping[str, float]) -> None:
        if not self.active:
            return
        for evolution in self.parameter_evolutions:
            evolution.validate(row)

    def evolve_row(self, row: Mapping[str, float], z: float) -> dict[str, float]:
        """Return a copy of ``row`` with configured parameters evolved to ``z``."""
        out = dict(row)
        if not self.active:
            return out
        for evolution in self.parameter_evolutions:
            out[evolution.parameter] = evolution.apply(
                row,
                z,
                reference_redshift=self.reference_redshift,
            )
        return out

    def describe(self) -> str:
        if not self.active:
            return "none"
        return ";".join(evolution.describe() for evolution in self.parameter_evolutions)


def evolve_power_law(value: float, z: float, power: float, *, reference_redshift: float = 0.0) -> float:
    """Backward-compatible helper for power-law evolution in ``1 + z``."""
    return evolve_value(value, z, power, kind="power_law", reference_redshift=reference_redshift)


def evolve_row_power_law(
    row: Mapping[str, float],
    z: float,
    parameter_powers: Mapping[str, PowerSpec],
    *,
    reference_redshift: float = 0.0,
) -> dict[str, float]:
    """Return a copy of ``row`` with selected parameters power-law evolved."""
    config = RedshiftEvolutionConfig.from_parameter_powers(
        parameter_powers,
        reference_redshift=reference_redshift,
    )
    return config.evolve_row(row, z)
