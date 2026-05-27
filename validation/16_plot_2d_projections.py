#!/usr/bin/env python
"""Make 2D projection diagnostics for the GWTC-4 posterior-predictive sampler.

What can be tested rigorously:
    * p(m1, q | Lambda) is analytic for the default model and can be plotted as
      equal-row averaged contours.
    * If a sample parquet from validation/08 exists, sampled 2D histograms can be
      overlaid against the analytic p(m1, q) contours.

What cannot usually be compared directly:
    The standard released ``rates_on_grids`` products are 1D marginal grids. They
    do not generally provide observed/theoretical 2D rate surfaces such as
    p(m1, q) or p(m1, chi_eff). Therefore this script treats 2D plots as
    model/sample self-consistency diagnostics unless a future release file exposes
    explicit multidimensional grids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from popsampler.analytic_marginals import analytic_mass_ratio_joint_for_rows, chi_eff_from_samples, normalize_pdf_2d
from popsampler.models import BBHDefaultModelConfig, GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift
from popsampler.popsummary_io import get_hyperparameter_samples


def load_or_choose_rows(path: Path, hyper_len: int, *, n_hyperrows: int, seed: int) -> np.ndarray:
    if path.exists():
        rows = np.load(path).astype(int)
        if len(rows) > n_hyperrows:
            return rows[:n_hyperrows]
        return rows
    rng = np.random.default_rng(seed)
    return rng.choice(hyper_len, size=n_hyperrows, replace=False)


def credible_levels_2d(pdf: np.ndarray, levels: tuple[float, ...] = (0.50, 0.90)) -> list[float]:
    """Return density thresholds enclosing requested posterior masses.

    This assumes a nearly uniform grid. The thresholds are used only for visual
    contouring, so exact cell-area weighting is not critical here.
    """
    flat = np.asarray(pdf, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat >= 0.0)]
    if len(flat) == 0 or np.sum(flat) <= 0.0:
        raise ValueError("Cannot compute contour levels from empty/zero PDF")
    order = np.argsort(flat)[::-1]
    sorted_pdf = flat[order]
    cdf = np.cumsum(sorted_pdf)
    cdf /= cdf[-1]
    thresholds = []
    for level in levels:
        idx = int(np.searchsorted(cdf, level, side="left"))
        thresholds.append(float(sorted_pdf[min(idx, len(sorted_pdf) - 1)]))
    return sorted(thresholds)


def plot_m1_q_analytic(
    h5: Path,
    rows: np.ndarray,
    outdir: Path,
    *,
    m1_bins: int,
    q_bins: int,
    m1_max: float,
) -> None:
    hyper = get_hyperparameter_samples(h5)
    model = GWTC4BrokenPowerLawTwoPeaksGaussianComponentSpinsPowerLawRedshift(
        config=BBHDefaultModelConfig(m1_max=m1_max, mass_grid_size=m1_bins, q_grid_size=q_bins, warn_if_unvalidated=False)
    )
    m1_grid = np.linspace(model.config.m1_min, model.config.m1_max, m1_bins)
    q_grid = np.linspace(model.config.q_min, model.config.q_max, q_bins)
    joint = analytic_mass_ratio_joint_for_rows(model, hyper, rows, m1_grid, q_grid)
    levels = credible_levels_2d(joint)

    np.savez(outdir / "m1_q_analytic_joint.npz", mass_1=m1_grid, mass_ratio=q_grid, pdf=joint)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    mesh = ax.pcolormesh(m1_grid, q_grid, joint.T, shading="auto")
    ax.contour(m1_grid, q_grid, joint.T, levels=levels, linewidths=1.5)
    ax.set_xlabel(r"$m_1\,[M_\odot]$")
    ax.set_ylabel(r"$q$")
    ax.set_title(r"analytic equal-row $p(m_1, q)$")
    fig.colorbar(mesh, ax=ax, label="normalized density")
    fig.tight_layout()
    fig.savefig(outdir / "m1_q_analytic_contours.png", dpi=200)
    plt.close(fig)


def plot_sample_projection(samples: pd.DataFrame, x: str, y: str, outpath: Path, *, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    hist = ax.hist2d(samples[x], samples[y], bins=bins, density=True)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"sample projection: {x} vs {y}")
    fig.colorbar(hist[3], ax=ax, label="normalized density")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def overlay_m1_q_samples(samples: pd.DataFrame, analytic_npz: Path, outpath: Path, *, bins: int) -> None:
    data = np.load(analytic_npz)
    m1_grid = data["mass_1"]
    q_grid = data["mass_ratio"]
    joint = normalize_pdf_2d(m1_grid, q_grid, data["pdf"])
    levels = credible_levels_2d(joint)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    hist = ax.hist2d(
        samples["mass_1_source"],
        samples["mass_ratio"],
        bins=bins,
        range=((float(m1_grid.min()), float(m1_grid.max())), (float(q_grid.min()), float(q_grid.max()))),
        density=True,
    )
    ax.contour(m1_grid, q_grid, joint.T, levels=levels, linewidths=1.5)
    ax.set_xlabel(r"$m_1\,[M_\odot]$")
    ax.set_ylabel(r"$q$")
    ax.set_title(r"sampled $(m_1,q)$ with analytic contours")
    fig.colorbar(hist[3], ax=ax, label="sample density")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    parser.add_argument("--samples", default="validation_outputs/joint_sampler_validation_samples.parquet")
    parser.add_argument("--rows", default="validation_outputs/joint_sampler_rows.npy")
    parser.add_argument("--outdir", default="validation_outputs/figures/2d")
    parser.add_argument("--n-hyperrows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--m1-bins", type=int, default=240)
    parser.add_argument("--q-bins", type=int, default=180)
    parser.add_argument("--sample-bins", type=int, default=120)
    parser.add_argument("--m1-max", type=float, default=120.0, help="Plot-range cap for m1 diagnostics")
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    samples_path = Path(args.samples).expanduser().resolve()
    rows_path = Path(args.rows).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hyper = get_hyperparameter_samples(h5)
    rows = load_or_choose_rows(rows_path, len(hyper), n_hyperrows=args.n_hyperrows, seed=args.seed)
    plot_m1_q_analytic(h5, rows, outdir, m1_bins=args.m1_bins, q_bins=args.q_bins, m1_max=args.m1_max)

    if samples_path.exists():
        samples = pd.read_parquet(samples_path)
        samples = samples.copy()
        samples["chi_eff"] = chi_eff_from_samples(samples)
        overlay_m1_q_samples(
            samples,
            outdir / "m1_q_analytic_joint.npz",
            outdir / "m1_q_samples_with_analytic_contours.png",
            bins=args.sample_bins,
        )
        plot_sample_projection(samples, "mass_1_source", "mass_2_source", outdir / "m1_m2_sample_projection.png", bins=args.sample_bins)
        plot_sample_projection(samples, "mass_1_source", "chi_eff", outdir / "m1_chi_eff_sample_projection.png", bins=args.sample_bins)
        plot_sample_projection(samples, "mass_ratio", "chi_eff", outdir / "q_chi_eff_sample_projection.png", bins=args.sample_bins)
    else:
        print(f"sample parquet not found, skipped sample overlays: {samples_path}")

    print(f"wrote 2D projection diagnostics to {outdir}")


if __name__ == "__main__":
    main()
