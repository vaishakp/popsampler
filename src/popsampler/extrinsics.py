"""Extrinsic-parameter samplers.

These are intentionally independent of the GWTC population hyperposterior unless
one later adds a detector-conditioned/detected-population model.
"""

from __future__ import annotations

import numpy as np


def sample_isotropic_extrinsics(
    n: int,
    *,
    rng: np.random.Generator | None = None,
    gps_start: float | None = None,
    duration: float | None = None,
) -> dict[str, np.ndarray]:
    """Sample sky, orientation, polarization, phase, and optionally time."""
    rng = np.random.default_rng() if rng is None else rng
    cos_theta_jn = rng.uniform(-1.0, 1.0, size=n)
    sin_dec = rng.uniform(-1.0, 1.0, size=n)
    out = {
        "ra": rng.uniform(0.0, 2.0 * np.pi, size=n),
        "dec": np.arcsin(sin_dec),
        "cos_theta_jn": cos_theta_jn,
        "theta_jn": np.arccos(cos_theta_jn),
        "psi": rng.uniform(0.0, np.pi, size=n),
        "phase": rng.uniform(0.0, 2.0 * np.pi, size=n),
    }
    if gps_start is not None and duration is not None:
        out["geocent_time"] = rng.uniform(float(gps_start), float(gps_start) + float(duration), size=n)
    return out
