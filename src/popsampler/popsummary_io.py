"""Thin wrappers around LVK popsummary HDF5 products.

The project intentionally treats popsummary files as the source of truth for
hyperposterior samples. Raw GWTC-4 HDF5 files may store
``posterior/hyperparameter_samples`` as an unnamed numeric array, so this module
uses the ``popsummary.PopulationResult`` metadata interface to recover physical
hyperparameter names when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

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


def _as_name_list(value) -> list[str]:
    """Convert popsummary metadata return values to a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def get_metadata(path: str | Path, field: str):
    """Return a metadata field from a popsummary file.

    Different popsummary versions accept either ``get_metadata(field='...')`` or
    ``get_metadata('...')``. This wrapper supports both.
    """
    result = open_result(path)
    try:
        return result.get_metadata(field=field)
    except TypeError:
        return result.get_metadata(field)


def get_hyperparameter_metadata(path: str | Path) -> list[str]:
    """Return hyperparameter names advertised by popsummary metadata.

    The raw HDF5 dataset may not have column attributes. The figure scripts in
    the GWTC-4 release use ``PopulationResult.get_metadata('hyperparameters')``
    and explicit ``get_hyperparameter_samples(hyperparameters=...)`` calls; this
    function follows that route.
    """
    try:
        names = _as_name_list(get_metadata(path, "hyperparameters"))
    except Exception:
        return []
    return names


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


def _to_dataframe(samples, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(samples, pd.DataFrame):
        df = samples.copy()
        if columns is not None and len(columns) == df.shape[1]:
            df.columns = list(columns)
        return df
    return pd.DataFrame(samples, columns=list(columns) if columns is not None else None)


def get_hyperparameter_samples(path: str | Path, hyperparameters: Iterable[str] | str | None = None) -> pd.DataFrame:
    """Load named hyperposterior samples from a popsummary file.

    Parameters
    ----------
    path:
        Popsummary HDF5 file.
    hyperparameters:
        Optional hyperparameter name or list of names. If omitted, the function
        first asks popsummary metadata for the complete hyperparameter-name list
        and then requests those names explicitly. This avoids returning unnamed
        numeric columns for GWTC-4 files whose raw HDF5 dataset has no column
        attributes.
    """
    result = open_result(path)

    if hyperparameters is None:
        names = get_hyperparameter_metadata(path)
        if names:
            samples = result.get_hyperparameter_samples(hyperparameters=names)
            return _to_dataframe(samples, columns=names)
        samples = result.get_hyperparameter_samples()
        return _to_dataframe(samples)

    if isinstance(hyperparameters, str):
        names = [hyperparameters]
    else:
        names = list(hyperparameters)
    samples = result.get_hyperparameter_samples(hyperparameters=names)
    return _to_dataframe(samples, columns=names)


def infer_hyperparameter_names(path: str | Path) -> list[str]:
    """Best-effort extraction of hyperparameter column names."""
    names = get_hyperparameter_metadata(path)
    if names:
        return names
    df = get_hyperparameter_samples(path)
    return [str(col) for col in df.columns]
