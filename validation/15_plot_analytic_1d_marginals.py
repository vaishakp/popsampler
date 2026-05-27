#!/usr/bin/env python
"""Plot analytic GWTC-4 1D marginals against released rate grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from popsampler.analytic_marginals import DEFAULT_1D_PARAMETERS, XLABELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves-dir", default="validation_outputs/analytic_1d_curves")
    parser.add_argument("--outdir", default="validation_outputs/figures/analytic_1d")
    parser.add_argument("--parameters", nargs="+", default=list(DEFAULT_1D_PARAMETERS))
    args = parser.parse_args()

    curves_dir = Path(args.curves_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for name in args.parameters:
        path = curves_dir / f"{name}_analytic_vs_grid_curve.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing curve file: {path}")
        curve = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(curve["x"], curve["released_grid_pdf"], linewidth=2.0, label="released grid")
        ax.plot(curve["x"], curve["analytic_pdf"], linestyle="--", linewidth=2.0, label="analytic model")
        ax.set_xlabel(XLABELS.get(name, name))
        ax.set_ylabel("normalized density")
        ax.set_title(f"{name}: analytic model vs released 1D grid")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}_analytic_vs_grid.png", dpi=200)
        plt.close(fig)

    print(f"wrote figures to {outdir}")


if __name__ == "__main__":
    main()
