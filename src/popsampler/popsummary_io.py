"""Thin wrappers around LVK popsummary HDF5 products.

The project intentionally treats popsummary files as the source of truth for
hyperposterior samples. This module avoids hard-coding model hyperparameter names
outside model-specific sampler classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd


class PopsummaryDependencyError(RuntimeError):
    """Raised when popsummary is required but unavailable."""


def _import_population_result():
    try:
        from popsummary.popresult import PopulationResult
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise PopsummaryDependencyError(
            "Could not import popsummary.popresult.PopulationResult. "
            "Install with `pip install .[gwtc4]` or install popsummary manually."
        ) from exc
    return PopulationResult


def open_result(path: str | Path):
    """Open a popsummary result object."""
    PopulationResult = _import_population_result()
    return PopulationResult(str(path))


def list_hdf5_tree(path: str | Path, *, max_datasets: int | None = None) -> list[dict]:
    """Return a compact listing of groups/datasets in an HDF5 file."""
    rows: list[dict] = []
    with h5py.File(path, "r") as h5:
        def visit(name: str, obj):
            if max_datasets is not None and len(rows) >= max_datasets:
                return
            rows.append(
                {
                    "name": name,
                    "type": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                    "shape": getattr(obj, "shape", None),
                    "dtype": str(getattr(obj, "dtype", "")),
                }
            )
        h5.visititems(visit)
    return rows


def get_hyperparameter_samples(path: str | Path, hyperparameters: Iterable[str] | None = None) -> pd.DataFrame:
    """Load hyperposterior samples from a popsummary file.

    Parameters
    ----------
    path:
        Popsummary HDF5 file.
    hyperparameters:
        Optional list of hyperparameters. If omitted, this asks popsummary for
        all available hyperparameters. The exact behavior depends on the
        popsummary version and file schema.
    """
    result = open_result(path)
    if hyperparameters is None:
        samples = result.get_hyperparameter_samples()
    else:
        samples = result.get_hyperparameter_samples(hyperparameters=list(hyperparameters))
    if isinstance(samples, pd.DataFrame):
        return samples
    return pd.DataFrame(samples)


def infer_hyperparameter_names(path: str | Path) -> list[str]:
    """Best-effort extraction of hyperparameter column names."""
    df = get_hyperparameter_samples(path)
    return list(df.columns)
