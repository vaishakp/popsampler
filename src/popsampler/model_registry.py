"""Model registry for user-selectable posterior-predictive samplers."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .models import (
    BBHDefaultModelConfig,
    BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift,
    GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift,
    ModelValidationError,
)
from .notch_mass import NotchMassSampler, NotchMassSamplerConfig, resolve_notch_hyperparameters


def _required_value(row: Mapping[str, float], name: str) -> float:
    if name not in row or pd.isna(row[name]):
        raise ModelValidationError(f"Missing required hyperparameter {name!r}")
    value = float(row[name])
    if not np.isfinite(value):
        raise ModelValidationError(f"Hyperparameter {name!r} is not finite: {value}")
    return value


class GWTC4NotchMassGaussianComponentSpinsPowerLawRedshift(
    GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift
):
    """Native Notch mass model with the existing spin and redshift blocks.

    The mass block is implemented inside popsampler. It does not call GWForge,
    gwpopulation, or another population-model package.
    """

    name = "BBHMassSpinRedshift_NativeNotchMass_GaussianComponentSpins_PowerLawRedshift"

    MASS_PARAMETERS = ("alpha_1", "alpha_dip", "alpha_2", "NSmin", "NSmax", "BHmin")
    REQUIRED_PARAMETERS = MASS_PARAMETERS + (
        "mu_chi",
        "sigma_chi",
        "mu_spin",
        "sigma_spin",
        "xi_spin",
    )

    def __init__(self, config: BBHDefaultModelConfig | None = None):
        self.config = BBHDefaultModelConfig() if config is None else config
        self.mass_sampler = NotchMassSampler(
            NotchMassSamplerConfig(
                m1_min=self.config.m1_min,
                m1_max=self.config.m1_max,
                q_min=self.config.q_min,
                q_max=self.config.q_max,
                mass_grid_size=self.config.mass_grid_size,
                q_grid_size=self.config.q_grid_size,
            )
        )
        self._warned = False

    def validate_hyperparameters(self, row: Mapping[str, float]) -> None:
        missing = [name for name in self.REQUIRED_PARAMETERS if name not in row or pd.isna(row[name])]
        if missing:
            raise ModelValidationError(f"Missing required hyperparameters for Notch model: {missing}")
        for name in self.REQUIRED_PARAMETERS:
            _required_value(row, name)
        self._validate_physical_constraints(row)
        self.redshift_rate.validate(row)
        self.redshift_evolution.validate(row)

    def _validate_physical_constraints(self, row: Mapping[str, float]) -> None:
        try:
            resolve_notch_hyperparameters(row, default_bh_max=self.config.m1_max)
        except ValueError as exc:
            raise ModelValidationError(str(exc)) from exc
        if _required_value(row, "sigma_chi") <= 0 or _required_value(row, "sigma_spin") <= 0:
            raise ModelValidationError("Gaussian spin parameters sigma_chi/sigma_spin must be positive")
        xi = _required_value(row, "xi_spin")
        if not (0.0 <= xi <= 1.0):
            raise ModelValidationError(f"Expected 0 <= xi_spin <= 1, got {xi}")

    def sample_masses(
        self,
        row: Mapping[str, float],
        n: int,
        *,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        return self.mass_sampler.sample(row, n, rng=rng)

    def _annotate_dataframe(self, df: pd.DataFrame, *, redshift_evolution_applied: bool) -> None:
        super()._annotate_dataframe(df, redshift_evolution_applied=redshift_evolution_applied)
        df["model_name"] = self.name
        df["sampler_mode"] = (
            "native_notch_joint_model_redshift_evolved"
            if redshift_evolution_applied
            else "native_notch_joint_model"
        )
        df["joint_sampler_validation_status"] = "native_notch_needs_rates_on_grids_validation"


MODEL_REGISTRY = {
    "bpl2peak": BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift,
    "notch": GWTC4NotchMassGaussianComponentSpinsPowerLawRedshift,
}


def get_bbh_model(name: str, config: BBHDefaultModelConfig | None = None):
    key = _normalize_model_name(name)
    return MODEL_REGISTRY[key](config)


def available_models() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def _normalize_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "bpl": "bpl2peak",
        "bpl2peak": "bpl2peak",
        "broken_power_law_two_peaks": "bpl2peak",
        "default": "bpl2peak",
        "notch": "notch",
        "notch_mass": "notch",
    }
    if key not in aliases:
        raise ValueError(f"Unknown model name {name!r}; choose one of {sorted(MODEL_REGISTRY)}")
    return aliases[key]
