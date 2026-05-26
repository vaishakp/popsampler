#!/usr/bin/env python
"""Plot 1D sampler/model comparisons against released GWTC-4 grids.

This script makes publication/debugging figures from the validation outputs. It
uses two different redshift objects deliberately:

* masses/spins/q: event-sample histograms are compared to released 1D grids;
* redshift: the released grid is compared to R(z) ∝ (1+z)^lambda, because the
  GWTC-4 redshift grid stores the comoving source-frame rate-density shape, not
  the detector-frame event-redshift distribution sampled by the catalog sampler.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from popsampler.grid_rates import load_rate_grid
from popsampler.popsummary_io import get_hyperparameter_samples


SAMPLE_PARAMETERS = ["mass_1", "mass_ratio", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
SAMPLE_COLUMNS = {
    "mass_1": "mass_1_source",
    "mass_ratio": "mass_ratio",
    "a_1": "a_1",
    "a_2": "a_2",
    "cos_tilt_1": "cos_tilt_1",
    "cos_tilt_2": "cos_tilt_2",
}
XLABELS = {
    "mass_1": r"$m_1\,[M_\odot]$",
    "mass_ratio": r"$q$",
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "cos_tilt_1": r"$\cos\theta_1$",
    "cos_tilt_2": r"$\cos\theta_2$",
    "redshift": r"$z$",
}


def normalized_pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.0, np.inf)
    norm = np.trapezoid(y, x)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Cannot normalize curve; integral={norm}")
    return y / norm


def grid_target_pdf(x: np.ndarray, rates: np.ndarray, *, target_weighting: str) -> np.ndarray:
    rates = np.clip(np.asarray(rates, dtype=float), 0.0, np.inf)
    if target_weighting == "rate_weighted":
        return normalized_pdf(x, rates.mean(axis=0))
    if target_weighting == "equal_row_pdf":
        norms = np.trapezoid(rates, x, axis=1)
        good = np.isfinite(norms) & (norms > 0)
        if not np.any(good):
            raise ValueError("No selected rows have positive finite grid integrals")
        pdfs = rates[good] / norms[good, None]
        return normalized_pdf(x, pdfs.mean(axis=0))
    raise ValueError(target_weighting)


def redshift_rate_density_model(z: np.ndarray, hyper: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    pdfs = []
    for idx in rows:
        lamb = float(hyper.iloc[int(idx)]["lamb"])
        pdfs.append(normalized_pdf(z, np.power(1.0 + z, lamb)))
    return normalized_pdf(z, np.mean(pdfs, axis=0))


def load_rows(path: Path, hyper_len: int, *, n_hyperrows: int, seed: int) -> np.ndarray:
    if path.exists():
        return np.load(path).astype(int)
    rng = np.random.default_rng(seed)
    return rng.choice(hyper_len, size=n_hyperrows, replace=False)


def plot_sample_parameter(
    samples: pd.DataFrame,
    h5: Path,
    rows: np.ndarray,
    name: str,
    outpath: Path,
    *,
    bins: int,
    target_weighting: str,
) -> None:
    grid = load_rate_grid(h5, name)
    x = grid.positions
    target = grid_target_pdf(x, grid.rates[rows], target_weighting=target_weighting)
    values = samples[SAMPLE_COLUMNS[name]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(values, bins=bins, range=(float(x.min()), float(x.max())), density=True, histtype="step", linewidth=1.8, label="sampler")
    ax.plot(x, target, linewidth=2.0, label=f"released grid ({target_weighting})")
    ax.set_xlabel(XLABELS.get(name, name))
    ax.set_ylabel("normalized density")
    ax.set_title(name)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_redshift_rate_density(
    hyper: pd.DataFrame,
    h5: Path,
    rows: np.ndarray,
    outpath: Path,
    *,
    target_weighting: str,
) -> None:
    grid = load_rate_grid(h5, "redshift")
    z = grid.positions
    target = grid_target_pdf(z, grid.rates[rows], target_weighting=target_weighting)
    model = redshift_rate_density_model(z, hyper, rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(z, target, linewidth=2.0, label=f"released grid ({target_weighting})")
    ax.plot(z, model, linestyle="--", linewidth=2.0, label=r"$R(z) \propto (1+z)^\lambda$")
    ax.set_xlabel(XLABELS["redshift"])
    ax.set_ylabel("normalized density")
    ax.set_title("redshift rate-density check")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--samples", default="validation_outputs/joint_sampler_validation_samples.parquet")
    parser.add_argument("--rows", default="validation_outputs/joint_sampler_rows.npy")
    parser.add_argument("--outdir", default="validation_outputs/figures")
    parser.add_argument("--n-hyperrows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bins", type=int, default=150)
    parser.add_argument(
        "--target-weighting",
        choices=["equal_row_pdf", "rate_weighted"],
        default="equal_row_pdf",
    )
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    samples_path = Path(args.samples).expanduser().resolve()
    rows_path = Path(args.rows).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(samples_path)
    hyper = get_hyperparameter_samples(h5)
    rows = load_rows(rows_path, len(hyper), n_hyperrows=args.n_hyperrows, seed=args.seed)

    for name in SAMPLE_PARAMETERS:
        plot_sample_parameter(
            samples,
            h5,
            rows,
            name,
            outdir / f"{name}_sample_vs_grid.png",
            bins=args.bins,
            target_weighting=args.target_weighting,
        )
    plot_redshift_rate_density(
        hyper,
        h5,
        rows,
        outdir / "redshift_rate_density_model_vs_grid.png",
        target_weighting=args.target_weighting,
    )

    print(f"wrote figures to {outdir}")


if __name__ == "__main__":
    main()
