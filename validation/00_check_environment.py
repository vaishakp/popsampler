#!/usr/bin/env python
"""Step 00: check the validation environment."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GWTC-4 popsummary HDF5 file")
    args = parser.parse_args()

    h5 = Path(args.h5).expanduser().resolve()
    print("# Environment")
    for package in ["popsampler", "numpy", "pandas", "h5py", "scipy", "astropy", "popsummary", "pyarrow"]:
        print(f"{package:12s}: {version(package)}")

    print("\n# File")
    print(f"h5: {h5}")
    print(f"exists: {h5.exists()}")
    if not h5.exists():
        raise FileNotFoundError(h5)

    from popsampler.grid_rates import list_rate_grid_names

    names = list_rate_grid_names(h5)
    print("\n# Rate grids")
    print(names)
    if not names:
        raise RuntimeError("No posterior/rates_on_grids entries found")


if __name__ == "__main__":
    main()
