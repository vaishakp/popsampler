"""Shared helpers for validation scripts.

Run these scripts from the repository root after installing the package, e.g.

    python -m pip install -e '.[gwtc4,parquet,dev]'

The default HDF5 path can be overridden with either `--h5` on each script or
with the environment variable `POPSAMPLER_GWTC4_H5`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_H5 = Path(
    "data/gwtc4/analyses_BBH/"
    "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5"
)
DEFAULT_OUTPUT_DIR = Path("validation_outputs")


def default_h5() -> Path:
    return Path(os.environ.get("POPSAMPLER_GWTC4_H5", DEFAULT_H5)).expanduser()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--h5",
        default=str(default_h5()),
        help="Path to the GWTC-4 popsummary HDF5 file. Can also set POPSAMPLER_GWTC4_H5.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for validation outputs.",
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    h5 = Path(args.h5).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return h5, output_dir


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run `popsampler-download-gwtc4 --cache-dir data/gwtc4` first, or pass --h5."
        )


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    # numpy.trapezoid exists in newer NumPy; trapz is kept as fallback.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def normalize_pdf(rate: np.ndarray, x: np.ndarray) -> np.ndarray:
    rate = np.asarray(rate, dtype=float)
    x = np.asarray(x, dtype=float)
    rate = np.clip(rate, 0.0, np.inf)
    norm = trapezoid(rate, x)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Rate/PDF has non-positive integral")
    return rate / norm


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")


def load_row_indices(output_dir: Path) -> np.ndarray:
    row_file = output_dir / "grid_marginal_rows.npy"
    if not row_file.exists():
        raise FileNotFoundError(
            f"Missing {row_file}. Run validation/03_generate_grid_marginal_samples.py first."
        )
    return np.load(row_file)
