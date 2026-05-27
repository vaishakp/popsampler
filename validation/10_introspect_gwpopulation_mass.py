#!/usr/bin/env python
"""Inspect the local gwpopulation mass-model API and call conventions."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np

from popsampler.popsummary_io import get_hyperparameter_samples
from popsampler.gwpopulation_mass import CANDIDATE_CLASS_NAMES, GWPopulationMassSampler


def safe_source(obj, max_lines: int = 80) -> list[str]:
    try:
        src = inspect.getsource(obj).splitlines()
    except Exception as exc:
        return [f"<source unavailable: {type(exc).__name__}: {exc}>"]
    if len(src) > max_lines:
        return src[:max_lines] + [f"... truncated after {max_lines} lines ..."]
    return src


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--outdir", default="validation_outputs")
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    import gwpopulation
    import gwpopulation.models.mass as mass

    h5 = Path(args.h5).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "gwpopulation_mass_api.txt"

    hyper = get_hyperparameter_samples(h5)
    row = hyper.iloc[args.row].to_dict()
    sampler = GWPopulationMassSampler()
    dataset = sampler._dataset(np.array([10.0, 20.0]), np.array([0.8, 0.5]))
    kwargs = sampler._row_to_kwargs(row)
    dataset_with_params = dict(dataset)
    dataset_with_params.update(kwargs)

    ctor_attempts = [
        ("default", {}),
        ("mmax_300", {"mmin": 2.0, "mmax": 300.0}),
        ("row_mmax", {"mmin": 2.0, "mmax": float(kwargs.get("mmax", 300.0))}),
        ("shape_tuple", {"mmin": 2.0, "mmax": float(kwargs.get("mmax", 300.0)), "normalization_shape": (400, 200)}),
    ]

    call_attempts = [
        ("model(dataset, **kwargs)", lambda model: model(dataset, **kwargs)),
        ("model(dataset_with_params)", lambda model: model(dataset_with_params)),
        ("model(dataset_with_params, **kwargs)", lambda model: model(dataset_with_params, **kwargs)),
        ("model(dataset, kwargs)", lambda model: model(dataset, kwargs)),
    ]

    lines: list[str] = []
    lines.append(f"gwpopulation module: {getattr(gwpopulation, '__file__', None)}")
    lines.append(f"gwpopulation version: {getattr(gwpopulation, '__version__', 'UNKNOWN')}")
    lines.append(f"mass module: {getattr(mass, '__file__', None)}")
    lines.append("")
    lines.append("top-level mass module functions/classes containing Mass, mass, peak, power, smooth:")
    names = [n for n in dir(mass) if any(s in n.lower() for s in ["mass", "peak", "power", "smooth"])]
    for n in names:
        obj = getattr(mass, n)
        try:
            sig = inspect.signature(obj)
        except Exception:
            sig = ""
        lines.append(f"  {n}: {type(obj)} {sig}")
    lines.append("")
    lines.append("row keys relevant to mass:")
    for key in ["alpha_1", "alpha_2", "beta", "break_mass", "delta_m_1", "delta_m_2", "lam_0", "lam_1", "lamb", "mlow_1", "mlow_2", "mmax", "mpp_1", "mpp_2", "sigpp_1", "sigpp_2", "mmin", "mmin_1", "mmin_2", "delta_m"]:
        if key in kwargs:
            lines.append(f"  {key} = {kwargs[key]}")
    lines.append("")
    lines.append(f"dataset keys: {sorted(dataset)}")
    lines.append(f"dataset_with_params has beta: {'beta' in dataset_with_params}")
    lines.append(f"dataset_with_params keys sample: {sorted(dataset_with_params)[:80]}")
    lines.append("")

    for name in CANDIDATE_CLASS_NAMES:
        cls = getattr(mass, name, None)
        if cls is None:
            lines.append(f"{name}: MISSING")
            continue
        lines.append(f"{name}: PRESENT")
        try:
            lines.append(f"  class signature: {inspect.signature(cls)}")
        except Exception as exc:
            lines.append(f"  class signature error: {exc}")
        lines.append("  class source excerpt:")
        for src_line in safe_source(cls, max_lines=70):
            lines.append(f"    {src_line}")

        for ctor_label, ctor_kwargs in ctor_attempts:
            lines.append(f"  constructor {ctor_label}: {ctor_kwargs}")
            try:
                model = cls(**ctor_kwargs)
                lines.append(f"    model type: {type(model)}")
                lines.append(f"    model call signature: {inspect.signature(model)}")
                if hasattr(model, "__dict__"):
                    simple_attrs = {
                        k: v for k, v in model.__dict__.items()
                        if isinstance(v, (int, float, str, bool, tuple, list))
                    }
                    lines.append(f"    simple attrs: {simple_attrs}")
            except Exception as exc:
                lines.append(f"    construct FAIL {type(exc).__name__}: {exc}")
                continue
            for label, func in call_attempts:
                try:
                    values = func(model)
                    arr = np.asarray(values, dtype=float)
                    lines.append(f"    {label}: OK shape={arr.shape} values={arr}")
                except Exception as exc:
                    lines.append(f"    {label}: FAIL {type(exc).__name__}: {exc}")
        lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"wrote {report_path}")
    print(report_path.read_text())


if __name__ == "__main__":
    main()
