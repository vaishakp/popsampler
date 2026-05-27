"""Native redshift-rate models for posterior-predictive sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .cosmology import DEFAULT_COSMOLOGY, differential_comoving_volume_full_sky


NumberOrColumn = float | str


@dataclass(frozen=True)
class RedshiftRateConfig:
    """Configuration for the source-frame merger-rate evolution R(z)."""

    model: str = "power_law"
    power_law_index: NumberOrColumn = "lamb"
    madau_alpha: NumberOrColumn | None = None
    madau_beta: NumberOrColumn | None = None
    madau_z_peak: NumberOrColumn | None = None

    def __post_init__(self) -> None:
        normalized = normalize_redshift_model_name(self.model)
        object.__setattr__(self, "model", normalized)

    def validate(self, row: Mapping[str, float]) -> None:
        self.resolve_parameters(row)

    def resolve_parameters(self, row: Mapping[str, float]) -> dict[str, float]:
        if self.model == "power_law":
            return {"lamb": _resolve_spec(row, self.power_law_index, "power_law_index")}
        if self.model == "madau_dickinson":
            alpha = _resolve_optional_spec(
                row,
                self.madau_alpha,
                aliases=("madau_alpha", "md_alpha", "alpha_z", "gamma"),
                label="madau_alpha",
            )
            beta = _resolve_optional_spec(
                row,
                self.madau_beta,
                aliases=("madau_beta", "md_beta", "beta_z", "kappa"),
                label="madau_beta",
            )
            z_peak = _resolve_optional_spec(
                row,
                self.madau_z_peak,
                aliases=("madau_z_peak", "md_z_peak", "z_peak", "zp", "z_p"),
                label="madau_z_peak",
            )
            if z_peak <= 0.0:
                raise ValueError(f"Madau-Dickinson z_peak must be positive, got {z_peak}")
            return {"alpha": alpha, "beta": beta, "z_peak": z_peak}
        raise ValueError(f"Unknown redshift model {self.model!r}")

    def describe(self) -> str:
        if self.model == "power_law":
            return f"power_law(index={self.power_law_index})"
        return (
            "madau_dickinson("
            f"alpha={self.madau_alpha}, beta={self.madau_beta}, z_peak={self.madau_z_peak})"
        )


def normalize_redshift_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "power_law": "power_law",
        "powerlaw": "power_law",
        "pl": "power_law",
        "default": "power_law",
        "madau_dickinson": "madau_dickinson",
        "madau_dickenson": "madau_dickinson",
        "madau": "madau_dickinson",
        "md": "madau_dickinson",
    }
    if key not in aliases:
        raise ValueError("Unknown redshift model " + repr(name) + "; choose power_law or madau_dickinson")
    return aliases[key]


def available_redshift_models() -> tuple[str, ...]:
    return ("power_law", "madau_dickinson")


def redshift_pdf(
    z: np.ndarray,
    row: Mapping[str, float],
    config: RedshiftRateConfig,
    *,
    cosmology=DEFAULT_COSMOLOGY,
) -> np.ndarray:
    """Unnormalized detected-source redshift PDF for the configured R(z)."""

    params = config.resolve_parameters(row)
    if config.model == "power_law":
        rate = power_law_rate(z, params["lamb"])
    elif config.model == "madau_dickinson":
        rate = madau_dickinson_rate(
            z,
            alpha=params["alpha"],
            beta=params["beta"],
            z_peak=params["z_peak"],
        )
    else:
        raise ValueError(f"Unknown redshift model {config.model!r}")
    z = np.asarray(z, dtype=float)
    pdf = rate * differential_comoving_volume_full_sky(z, cosmology=cosmology) / (1.0 + z)
    return np.clip(pdf, 0.0, np.inf)


def power_law_rate(z: np.ndarray, lamb: float) -> np.ndarray:
    """Source-frame merger-rate evolution R(z) proportional to (1+z)^lamb."""

    z = np.asarray(z, dtype=float)
    return np.power(1.0 + z, float(lamb))


def madau_dickinson_rate(z: np.ndarray, *, alpha: float, beta: float, z_peak: float) -> np.ndarray:
    """Madau-Dickinson-like source-frame merger-rate evolution.

    The parameterization is

        R(z) proportional to (1 + z)^alpha
            / {1 + [((1 + z) / (1 + z_peak))]^(alpha + beta)}.

    At z << z_peak it behaves approximately as (1+z)^alpha; at high redshift it
    falls approximately as (1+z)^(-beta).  The original star-formation-rate
    shape corresponds roughly to alpha=2.7, beta=2.9, z_peak=1.9.
    """

    z = np.asarray(z, dtype=float)
    if z_peak <= 0.0:
        raise ValueError(f"z_peak must be positive, got {z_peak}")
    one_plus_z = 1.0 + z
    turnover = (1.0 + float(z_peak))
    exponent = float(alpha) + float(beta)
    denominator = 1.0 + np.power(one_plus_z / turnover, exponent)
    return np.clip(np.power(one_plus_z, float(alpha)) / denominator, 0.0, np.inf)


def _resolve_optional_spec(
    row: Mapping[str, float],
    spec: NumberOrColumn | None,
    *,
    aliases: tuple[str, ...],
    label: str,
) -> float:
    if spec is not None:
        return _resolve_spec(row, spec, label)
    for name in aliases:
        if name in row and not pd.isna(row[name]):
            return _finite_float(row[name], name)
    raise ValueError(
        f"Missing required {label} for Madau-Dickinson redshift model. "
        f"Set it in [redshift] or provide one of these hyperposterior columns: {aliases}."
    )


def _resolve_spec(row: Mapping[str, float], spec: NumberOrColumn, label: str) -> float:
    if isinstance(spec, str):
        try:
            return _finite_float(float(spec), label)
        except ValueError:
            pass
        if spec not in row or pd.isna(row[spec]):
            raise ValueError(f"Missing redshift parameter column {spec!r} for {label}")
        return _finite_float(row[spec], spec)
    return _finite_float(spec, label)


def _finite_float(value: object, label: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"Redshift parameter {label!r} is not finite: {out}")
    return out
