#!/usr/bin/env python
"""Step 04: check whether named GWTC-4 hyperparameter rows are usable.

This script intentionally separates two questions:

1. Did popsummary recover named hyperparameter columns from the release metadata?
2. Do the rows contain the physical hyperparameters needed by the default
   GWTC-4 BBH mass-spin-redshift model?

Passing this script means that hyperposterior rows are named and internally
consistent. It does *not* mean that the analytic joint sampler is validated; that
requires implementing or wrapping the exact GWTC-4 model formula and validating
its 1D projections against the released rates_on_grids products.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from popsampler.popsummary_io import get_hyperparameter_samples


MODEL_PARAMETER_GROUPS = {
    "mass": [
        "alpha_1",
        "alpha_2",
        "beta",
        "break_mass",
        "delta_m_1",
        "delta_m_2",
        "lam_0",
        "lam_1",
        "mlow_1",
        "mlow_2",
        "mmax",
        "mpp_1",
        "mpp_2",
        "sigpp_1",
        "sigpp_2",
    ],
    "spin": [
        "alpha_chi",
        "beta_chi",
        "amax",
        "mu_chi",
        "sigma_chi",
        "mu_spin",
        "sigma_spin",
        "xi_spin",
    ],
    "redshift_rate_selection": [
        "lamb",
        "rate",
        "log_10_rate",
        "selection",
        "surveyed_hypervolume",
        "pdet_n_effective",
        "selection_variance",
    ],
}

DIAGNOSTIC_PREFIXES = ("ln_bf_", "var_")
DIAGNOSTIC_COLUMNS = {"log_likelihood", "log_prior", "variance"}


def columns_are_named(columns: pd.Index) -> bool:
    if len(columns) == 0:
        return False
    return not all(isinstance(c, int) for c in columns)


def finite_fraction(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        if col in df:
            out[col] = float(np.isfinite(df[col].to_numpy(dtype=float)).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    parser.add_argument("--outdir", default="validation_outputs", help="Directory for validation outputs")
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hyper = get_hyperparameter_samples(h5)
    report_path = outdir / "analytic_sampler_readiness.txt"

    all_required = [p for group in MODEL_PARAMETER_GROUPS.values() for p in group]
    missing = [p for p in all_required if p not in hyper.columns]
    present = [p for p in all_required if p in hyper.columns]
    diagnostic_cols = [
        col for col in hyper.columns
        if col in DIAGNOSTIC_COLUMNS or any(str(col).startswith(prefix) for prefix in DIAGNOSTIC_PREFIXES)
    ]
    unknown_cols = [
        col for col in hyper.columns
        if col not in all_required and col not in diagnostic_cols
    ]

    with report_path.open("w") as fp:
        fp.write(f"hyperparameter_samples_shape: {hyper.shape}\n")
        fp.write(f"first_columns: {list(hyper.columns[:20])}\n")
        fp.write(f"columns_are_named: {columns_are_named(hyper.columns)}\n")

        if not columns_are_named(hyper.columns):
            fp.write("\nSTATUS: NOT_READY\n")
            fp.write(
                "Reason: popsummary returned numeric hyperparameter columns. The sampler\n"
                "needs named physical hyperparameters for covariance-preserving draws.\n"
            )
        else:
            fp.write("\nSTATUS: NAMED_COLUMNS_RECOVERED\n")

            fp.write("\n# Expected physical model parameters\n")
            for group, params in MODEL_PARAMETER_GROUPS.items():
                group_missing = [p for p in params if p not in hyper.columns]
                group_present = [p for p in params if p in hyper.columns]
                fp.write(f"\n[{group}]\n")
                fp.write(f"present ({len(group_present)}): {group_present}\n")
                fp.write(f"missing ({len(group_missing)}): {group_missing}\n")
                fp.write(f"finite_fraction: {finite_fraction(hyper, group_present)}\n")

            fp.write("\n# Diagnostic/non-model columns\n")
            fp.write(f"diagnostic_count: {len(diagnostic_cols)}\n")
            fp.write(f"diagnostic_examples: {diagnostic_cols[:20]}\n")
            fp.write(f"unknown_or_unclassified_count: {len(unknown_cols)}\n")
            fp.write(f"unknown_or_unclassified: {unknown_cols}\n")

            if missing:
                fp.write("\nSTATUS: NAMED_BUT_REQUIRED_MODEL_PARAMETERS_MISSING\n")
                fp.write(f"missing_required: {missing}\n")
            else:
                fp.write("\nSTATUS: READY_FOR_EXACT_MODEL_IMPLEMENTATION\n")
                fp.write(
                    "Named hyperposterior rows contain the expected physical parameters.\n"
                    "The next blocker is not metadata; it is implementing or wrapping the\n"
                    "exact GWTC-4 BrokenPowerLawTwoPeaks + GaussianComponentSpins +\n"
                    "PowerLawRedshift model and validating its 1D projections against\n"
                    "posterior/rates_on_grids.\n"
                )

    print(f"wrote {report_path}")
    print(report_path.read_text())


if __name__ == "__main__":
    main()
