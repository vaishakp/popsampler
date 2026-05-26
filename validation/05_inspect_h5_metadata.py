#!/usr/bin/env python
"""Step 05: inspect HDF5 metadata relevant to joint/covariance-preserving sampling.

This script is intentionally verbose. It searches for dataset attributes, string
objects, and structured dtypes that may encode the mapping between numeric
hyperparameter columns and physical parameter names.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


KEYWORDS = (
    "hyper",
    "parameter",
    "param",
    "name",
    "label",
    "column",
    "model",
    "mass",
    "spin",
    "redshift",
    "rate",
    "sample",
)


def preview(value: Any, max_chars: int = 1000) -> str:
    text = repr(value)
    if len(text) > max_chars:
        text = text[:max_chars] + " ... <truncated>"
    return text


def decode_array(arr: np.ndarray, max_items: int = 20) -> str:
    flat = np.asarray(arr).reshape(-1)
    items = []
    for item in flat[:max_items]:
        if isinstance(item, bytes):
            try:
                items.append(item.decode())
            except Exception:
                items.append(repr(item))
        else:
            items.append(repr(item))
    suffix = "" if flat.size <= max_items else f" ... ({flat.size} total)"
    return ", ".join(items) + suffix


def looks_relevant(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in KEYWORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to popsummary HDF5 file")
    parser.add_argument("--outdir", default="validation_outputs", help="Directory for outputs")
    parser.add_argument("--max-small-dataset-size", type=int, default=5000)
    args = parser.parse_args()

    h5_path = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "h5_metadata_report.txt"
    csv_path = outdir / "h5_datasets.csv"

    dataset_rows = []

    with h5py.File(h5_path, "r") as h5, report_path.open("w") as fp:
        fp.write(f"# HDF5 metadata report\nfile: {h5_path}\n\n")

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            obj_type = "dataset" if isinstance(obj, h5py.Dataset) else "group"
            shape = getattr(obj, "shape", None)
            dtype = str(getattr(obj, "dtype", ""))
            dataset_rows.append({"name": name, "type": obj_type, "shape": str(shape), "dtype": dtype})

            if obj.attrs:
                fp.write(f"\n## attrs: {name or '/'} ({obj_type})\n")
                for k, v in obj.attrs.items():
                    fp.write(f"{k}: {preview(v)}\n")

            if isinstance(obj, h5py.Dataset):
                dtype_obj = obj.dtype
                is_stringy = dtype_obj.kind in {"S", "U", "O"}
                is_structured = dtype_obj.fields is not None
                small = obj.size is not None and obj.size <= args.max_small_dataset_size
                if looks_relevant(name) or is_stringy or is_structured or small:
                    fp.write(f"\n## dataset: {name}\n")
                    fp.write(f"shape: {obj.shape}\n")
                    fp.write(f"dtype: {obj.dtype}\n")
                    if is_structured:
                        fp.write(f"structured_fields: {list(dtype_obj.fields or [])}\n")
                    if is_stringy or small or is_structured:
                        try:
                            arr = obj[()]
                            fp.write(f"preview: {decode_array(arr)}\n")
                        except Exception as exc:
                            fp.write(f"preview_error: {exc}\n")

        h5.visititems(visit)

        # Focused known paths.
        fp.write("\n# Focused known paths\n")
        for path in [
            "posterior/hyperparameter_samples",
            "posterior/reweighted_event_samples",
            "posterior/reweighted_injections",
            "posterior/rates_on_grids/mass_1/positions",
            "posterior/rates_on_grids/mass_1/rates",
        ]:
            if path in h5:
                obj = h5[path]
                fp.write(f"\n## {path}\n")
                fp.write(f"shape: {getattr(obj, 'shape', None)}\n")
                fp.write(f"dtype: {getattr(obj, 'dtype', None)}\n")
                fp.write(f"attrs: {dict(obj.attrs)}\n")

    pd.DataFrame(dataset_rows).to_csv(csv_path, index=False)
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
