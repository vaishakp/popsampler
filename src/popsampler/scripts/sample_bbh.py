"""Draw BBH posterior-predictive samples from a popsummary file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from popsampler.config import merge_cli_config, read_sampler_config, require_complete_run_config
from popsampler.model_registry import available_models, get_bbh_model
from popsampler.popsummary_io import get_hyperparameter_samples
from popsampler.posterior_predictive import PosteriorPredictiveSampler
from popsampler.redshift_evolution import ParameterEvolution


def write_table(df, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(output, index=False)
    elif suffix in {".h5", ".hdf5"}:
        df.to_hdf(output, key="samples", mode="w")
    else:
        raise ValueError("Output suffix must be .csv, .parquet/.pq, or .h5/.hdf5")


def parse_evolution_spec(spec: str) -> ParameterEvolution:
    """Parse PARAM:KIND:COEFF[:CLIP_MIN:CLIP_MAX].

    COEFF may be a literal number or the name of a hyperposterior column. Examples:

    - mpp_2:power_law:gamma_mpp_2
    - beta:linear_log1pz:-0.5
    - xi_spin:linear_z:gamma_xi_spin:0:1
    """
    parts = spec.split(":")
    if len(parts) not in {3, 5}:
        raise ValueError(
            "Evolution specs must have form PARAM:KIND:COEFF or "
            "PARAM:KIND:COEFF:CLIP_MIN:CLIP_MAX"
        )
    parameter, kind, coefficient = parts[:3]
    if kind not in {"power_law", "linear_z", "linear_log1pz"}:
        raise ValueError(f"Unknown evolution kind {kind!r}")
    try:
        coefficient_value: str | float = float(coefficient)
    except ValueError:
        coefficient_value = coefficient

    clip_min = clip_max = None
    if len(parts) == 5:
        clip_min = None if parts[3].lower() == "none" else float(parts[3])
        clip_max = None if parts[4].lower() == "none" else float(parts[4])
    return ParameterEvolution(
        parameter=parameter,
        kind=kind,  # type: ignore[arg-type]
        coefficient=coefficient_value,
        clip_min=clip_min,
        clip_max=clip_max,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("popsummary_file", nargs="?", help="Popsummary HDF5 file")
    parser.add_argument("--config", help="INI-style sampler configuration file")
    parser.add_argument("--model", choices=available_models(), help="Mass model to use")
    parser.add_argument("--n-events", type=int, help="Number of source events to draw")
    parser.add_argument("--output", help="Output .csv, .parquet, or .h5 file")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Events per hyperposterior row")
    parser.add_argument("--no-extrinsics", action="store_true")
    parser.add_argument(
        "--redshift-evolve",
        action="append",
        default=[],
        metavar="PARAM:KIND:COEFF[:CLIP_MIN:CLIP_MAX]",
        help=(
            "Opt-in redshift evolution for one hyperparameter. KIND is one of "
            "power_law, linear_z, linear_log1pz. COEFF can be a number or a "
            "hyperposterior-column name. May be repeated."
        ),
    )
    parser.add_argument(
        "--redshift-evolution-reference-z",
        type=float,
        default=None,
        help="Reference redshift z_ref for --redshift-evolve specs.",
    )
    args = parser.parse_args(argv)

    file_config = read_sampler_config(args.config) if args.config else None
    evolutions = tuple(parse_evolution_spec(spec) for spec in args.redshift_evolve)
    run_config = merge_cli_config(
        file_config,
        popsummary_file=args.popsummary_file,
        output=args.output,
        n_events=args.n_events,
        seed=args.seed,
        batch_size=args.batch_size,
        include_extrinsics=False if args.no_extrinsics else None,
        model=args.model,
        redshift_evolutions=evolutions,
        redshift_evolution_reference_z=args.redshift_evolution_reference_z,
    )
    require_complete_run_config(run_config)

    rng = np.random.default_rng(run_config.seed)
    hyper = get_hyperparameter_samples(run_config.popsummary_file)  # type: ignore[arg-type]
    model = get_bbh_model(run_config.model, run_config.model_config)
    sampler = PosteriorPredictiveSampler(hyperposterior=hyper, model=model)
    samples = sampler.sample(
        run_config.n_events,  # type: ignore[arg-type]
        rng=rng,
        batch_size=run_config.batch_size,
        include_extrinsics=run_config.include_extrinsics,
    )
    write_table(samples, Path(run_config.output))  # type: ignore[arg-type]
    print(f"wrote {len(samples)} samples to {run_config.output}")
    print(f"model: {run_config.model}")
    if run_config.model_config.redshift_evolution.active:
        print(f"redshift_evolution: {run_config.model_config.redshift_evolution.describe()}")


if __name__ == "__main__":
    main()
