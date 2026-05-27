"""Plot released one-dimensional popsummary rate grids.

This command is for inspection and validation of released 1D marginal products.
It is not an event sampler.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from popsampler.grid_rates import list_rate_grid_names, load_rate_grid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("popsummary_file", help="GWTC-4 popsummary HDF5 file")
    parser.add_argument("--parameters", nargs="+", default=None, help="Rate-grid names to plot. Defaults to all available grids.")
    parser.add_argument("--output-dir", default="grid_marginal_plots", help="Directory for PNG outputs")
    parser.add_argument("--quantiles", nargs=2, type=float, default=[0.05, 0.95], metavar=("LOW", "HIGH"))
    parser.add_argument("--density", action="store_true", help="Normalize rate curves to unit area before plotting")
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt

    h5 = Path(args.popsummary_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    available = list_rate_grid_names(h5)
    names = available if args.parameters is None else args.parameters
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"Unknown rate-grid names {missing}. Available: {available}")

    for name in names:
        grid = load_rate_grid(h5, name)
        x = grid.positions
        curves = grid.rates.copy()
        ylabel = "rate"
        if args.density:
            norms = np.trapz(np.clip(curves, 0.0, np.inf), x, axis=1)
            good = np.isfinite(norms) & (norms > 0)
            curves = curves[good] / norms[good, None]
            ylabel = "normalized density"

        mean = curves.mean(axis=0)
        low = np.quantile(curves, args.quantiles[0], axis=0)
        high = np.quantile(curves, args.quantiles[1], axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.fill_between(x, low, high, alpha=0.25, label=f"{args.quantiles[0]:.2f}-{args.quantiles[1]:.2f}")
        ax.plot(x, mean, label="posterior mean")
        ax.set_xlabel(name)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Released 1D marginal: {name}")
        ax.legend(frameon=False)
        fig.tight_layout()
        outfile = output_dir / f"{name}.png"
        fig.savefig(outfile, dpi=200)
        plt.close(fig)
        print(f"wrote {outfile}")


if __name__ == "__main__":
    main()
