#!/usr/bin/env python
"""Step 04: check whether the analytic hyperparameter sampler can be used."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from popsampler.models import BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift
from popsampler.popsummary_io import get_hyperparameter_samples


def columns_are_named(columns: pd.Index) -> bool:
    if len(columns) == 0:
        return False
    return not all(isinstance(c, int) for c in columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    parser.add_argument("--outdir", default="validation_outputs", help="Directory for validation outputs")
    parser.add_argument("--n-check", type=int, default=5, help="Number of rows to test if columns are named")
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hyper = get_hyperparameter_samples(h5)
    report_path = outdir / "analytic_sampler_readiness.txt"
    model = BrokenPowerLawTwoPeaksGaussianSpinsPowerLawRedshift()

    with report_path.open("w") as fp:
        fp.write(f"hyperparameter_samples_shape: {hyper.shape}\n")
        fp.write(f"first_columns: {list(hyper.columns[:20])}\n")
        fp.write(f"columns_are_named: {columns_are_named(hyper.columns)}\n")

        if not columns_are_named(hyper.columns):
            fp.write("\nSTATUS: NOT_READY\n")
            fp.write(
                "Reason: popsummary returned numeric hyperparameter columns. The analytic sampler\n"
                "needs a mapping from the 339 numeric columns to physical hyperparameter names\n"
                "such as alpha, beta, mmin, mmax, lambda_z, etc.\n"
            )
        else:
            fp.write("\nSTATUS: COLUMNS_NAMED\n")
            failures = []
            for idx in range(min(args.n_check, len(hyper))):
                row = hyper.iloc[idx].to_dict()
                try:
                    model.validate_hyperparameters(row)
                    fp.write(f"row {idx}: OK\n")
                except Exception as exc:
                    failures.append((idx, str(exc)))
                    fp.write(f"row {idx}: FAIL: {exc}\n")
            if failures:
                fp.write("\nSTATUS: NAMED_BUT_MODEL_ALIASES_INCOMPLETE\n")
            else:
                fp.write("\nSTATUS: READY_FOR_ANALYTIC_SAMPLER_SMOKE_TEST\n")

    print(f"wrote {report_path}")
    print(report_path.read_text())


if __name__ == "__main__":
    main()
