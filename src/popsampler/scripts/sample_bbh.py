"""Draw GWTC-4.0-style BBH posterior-predictive samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from popsampler.models import BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift
from popsampler.popsummary_io import get_hyperparameter_samples
from popsampler.posterior_predictive import PosteriorPredictiveSampler


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("popsummary_file", help="GWTC-4.0 popsummary HDF5 file")
    parser.add_argument("--n-events", type=int, required=True, help="Number of source events to draw")
    parser.add_argument("--output", required=True, help="Output .csv, .parquet, or .h5 file")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Events per hyperposterior row")
    parser.add_argument("--no-extrinsics", action="store_true")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    hyper = get_hyperparameter_samples(args.popsummary_file)
    model = BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift()
    sampler = PosteriorPredictiveSampler(hyperposterior=hyper, model=model)
    samples = sampler.sample(
        args.n_events,
        rng=rng,
        batch_size=args.batch_size,
        include_extrinsics=not args.no_extrinsics,
    )
    write_table(samples, Path(args.output))
    print(f"wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
