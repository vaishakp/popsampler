"""INI-style configuration helpers for popsampler command-line tools."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import BBHDefaultModelConfig
from .redshift_evolution import ParameterEvolution, RedshiftEvolutionConfig


@dataclass(frozen=True)
class SamplerRunConfig:
    """Resolved configuration for one posterior-predictive sampler run."""

    popsummary_file: str | None = None
    output: str | None = None
    n_events: int | None = None
    seed: int | None = None
    batch_size: int = 1
    include_extrinsics: bool = True
    model: str = "bpl2peak"
    model_config: BBHDefaultModelConfig = field(default_factory=BBHDefaultModelConfig)


def read_sampler_config(path: str | Path) -> SamplerRunConfig:
    """Read an INI-style sampler config.

    Supported sections are ``[run]``, ``[model]``, ``[grid]``, and
    ``[redshift_evolution]``. Command-line arguments can still override the
    returned values in the CLI layer.
    """

    parser = configparser.ConfigParser()
    parser.optionxform = str
    read_files = parser.read(path)
    if not read_files:
        raise FileNotFoundError(f"Could not read config file {path!s}")

    run = parser["run"] if parser.has_section("run") else {}
    model = parser["model"] if parser.has_section("model") else {}
    grid = parser["grid"] if parser.has_section("grid") else {}
    redshift = parser["redshift_evolution"] if parser.has_section("redshift_evolution") else {}

    evolution_config = _parse_redshift_evolution(redshift)
    model_config = BBHDefaultModelConfig(
        m1_min=_get_float(grid, "m1_min", BBHDefaultModelConfig.m1_min),
        m1_max=_get_float(grid, "m1_max", BBHDefaultModelConfig.m1_max),
        q_min=_get_float(grid, "q_min", BBHDefaultModelConfig.q_min),
        q_max=_get_float(grid, "q_max", BBHDefaultModelConfig.q_max),
        z_min=_get_float(grid, "z_min", BBHDefaultModelConfig.z_min),
        z_max=_get_float(grid, "z_max", BBHDefaultModelConfig.z_max),
        mass_grid_size=_get_int(grid, "mass_grid_size", BBHDefaultModelConfig.mass_grid_size),
        q_grid_size=_get_int(grid, "q_grid_size", BBHDefaultModelConfig.q_grid_size),
        z_grid_size=_get_int(grid, "z_grid_size", BBHDefaultModelConfig.z_grid_size),
        spin_grid_size=_get_int(grid, "spin_grid_size", BBHDefaultModelConfig.spin_grid_size),
        warn_if_unvalidated=_get_bool(model, "warn_if_unvalidated", True),
        redshift_evolution=evolution_config,
    )

    return SamplerRunConfig(
        popsummary_file=_get_optional_str(run, "popsummary_file"),
        output=_get_optional_str(run, "output"),
        n_events=_get_optional_int(run, "n_events"),
        seed=_get_optional_int(run, "seed"),
        batch_size=_get_int(run, "batch_size", 1),
        include_extrinsics=not _get_bool(run, "no_extrinsics", False),
        model=_normalize_model_name(_get_str(model, "name", "bpl2peak")),
        model_config=model_config,
    )


def merge_cli_config(
    config: SamplerRunConfig | None,
    *,
    popsummary_file: str | None,
    output: str | None,
    n_events: int | None,
    seed: int | None,
    batch_size: int | None,
    include_extrinsics: bool | None,
    model: str | None,
    redshift_evolutions: tuple[ParameterEvolution, ...] = (),
    redshift_evolution_reference_z: float | None = None,
) -> SamplerRunConfig:
    """Merge optional CLI overrides onto an optional INI config."""

    base = SamplerRunConfig() if config is None else config
    model_config = base.model_config
    if redshift_evolutions:
        model_config = BBHDefaultModelConfig(
            m1_min=model_config.m1_min,
            m1_max=model_config.m1_max,
            q_min=model_config.q_min,
            q_max=model_config.q_max,
            z_min=model_config.z_min,
            z_max=model_config.z_max,
            mass_grid_size=model_config.mass_grid_size,
            q_grid_size=model_config.q_grid_size,
            z_grid_size=model_config.z_grid_size,
            spin_grid_size=model_config.spin_grid_size,
            cosmology=model_config.cosmology,
            warn_if_unvalidated=model_config.warn_if_unvalidated,
            redshift_evolution=RedshiftEvolutionConfig(
                enabled=True,
                parameter_evolutions=redshift_evolutions,
                reference_redshift=(
                    model_config.redshift_evolution.reference_redshift
                    if redshift_evolution_reference_z is None
                    else redshift_evolution_reference_z
                ),
            ),
        )
    elif redshift_evolution_reference_z is not None and model_config.redshift_evolution.active:
        model_config = _replace_redshift_reference(model_config, redshift_evolution_reference_z)

    return SamplerRunConfig(
        popsummary_file=popsummary_file or base.popsummary_file,
        output=output or base.output,
        n_events=n_events if n_events is not None else base.n_events,
        seed=seed if seed is not None else base.seed,
        batch_size=batch_size if batch_size is not None else base.batch_size,
        include_extrinsics=(include_extrinsics if include_extrinsics is not None else base.include_extrinsics),
        model=_normalize_model_name(model or base.model),
        model_config=model_config,
    )


def require_complete_run_config(config: SamplerRunConfig) -> None:
    """Raise a clear error if required run fields are absent."""

    missing: list[str] = []
    if not config.popsummary_file:
        missing.append("popsummary_file")
    if not config.output:
        missing.append("output")
    if config.n_events is None:
        missing.append("n_events")
    if missing:
        raise ValueError(
            "Missing required run settings: "
            + ", ".join(missing)
            + ". Provide them in [run] or through CLI arguments."
        )


def _replace_redshift_reference(config: BBHDefaultModelConfig, reference_z: float) -> BBHDefaultModelConfig:
    evolution = RedshiftEvolutionConfig(
        enabled=config.redshift_evolution.enabled,
        parameter_evolutions=config.redshift_evolution.parameter_evolutions,
        reference_redshift=reference_z,
    )
    return BBHDefaultModelConfig(
        m1_min=config.m1_min,
        m1_max=config.m1_max,
        q_min=config.q_min,
        q_max=config.q_max,
        z_min=config.z_min,
        z_max=config.z_max,
        mass_grid_size=config.mass_grid_size,
        q_grid_size=config.q_grid_size,
        z_grid_size=config.z_grid_size,
        spin_grid_size=config.spin_grid_size,
        cosmology=config.cosmology,
        warn_if_unvalidated=config.warn_if_unvalidated,
        redshift_evolution=evolution,
    )


def _parse_redshift_evolution(section: Any) -> RedshiftEvolutionConfig:
    if not section:
        return RedshiftEvolutionConfig.disabled()
    enabled = _get_bool(section, "enabled", False)
    reference_z = _get_float(section, "reference_z", 0.0)
    evolutions: list[ParameterEvolution] = []
    for key, value in section.items():
        if key in {"enabled", "reference_z"}:
            continue
        evolutions.append(_parse_evolution_assignment(key, value))
    return RedshiftEvolutionConfig(
        enabled=enabled or bool(evolutions),
        parameter_evolutions=tuple(evolutions),
        reference_redshift=reference_z,
    )


def _parse_evolution_assignment(parameter: str, value: str) -> ParameterEvolution:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) not in {2, 4}:
        raise ValueError(
            "Redshift evolution entries must be KIND, COEFF or "
            "KIND, COEFF, CLIP_MIN, CLIP_MAX"
        )
    kind, coefficient = parts[:2]
    try:
        coefficient_value: str | float = float(coefficient)
    except ValueError:
        coefficient_value = coefficient
    clip_min = clip_max = None
    if len(parts) == 4:
        clip_min = None if parts[2].lower() == "none" else float(parts[2])
        clip_max = None if parts[3].lower() == "none" else float(parts[3])
    return ParameterEvolution(
        parameter=parameter,
        kind=kind,  # type: ignore[arg-type]
        coefficient=coefficient_value,
        clip_min=clip_min,
        clip_max=clip_max,
    )


def _normalize_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "bpl": "bpl2peak",
        "bpl2peak": "bpl2peak",
        "broken_power_law_two_peaks": "bpl2peak",
        "default": "bpl2peak",
        "notch": "notch",
        "notch_mass": "notch",
        "notch_filter": "notch",
    }
    if key not in aliases:
        raise ValueError(f"Unknown model name {name!r}; choose one of bpl2peak, notch")
    return aliases[key]


def _get_str(section: Any, name: str, default: str) -> str:
    return str(section[name]).strip() if name in section else default


def _get_optional_str(section: Any, name: str) -> str | None:
    if name not in section:
        return None
    value = str(section[name]).strip()
    return value or None


def _get_optional_int(section: Any, name: str) -> int | None:
    return int(section[name]) if name in section and str(section[name]).strip() else None


def _get_int(section: Any, name: str, default: int) -> int:
    return int(section[name]) if name in section else int(default)


def _get_float(section: Any, name: str, default: float) -> float:
    return float(section[name]) if name in section else float(default)


def _get_bool(section: Any, name: str, default: bool) -> bool:
    if name not in section:
        return bool(default)
    value = str(section[name]).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value for {name!r}: {section[name]!r}")
