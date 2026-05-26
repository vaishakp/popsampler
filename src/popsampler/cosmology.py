"""Cosmological helper functions for population sampling."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18, z_at_value


DEFAULT_COSMOLOGY = Planck18


def luminosity_distance_from_redshift(z: np.ndarray, cosmology=DEFAULT_COSMOLOGY) -> np.ndarray:
    return cosmology.luminosity_distance(z).to_value(u.Mpc)


def redshift_from_luminosity_distance(distance_mpc: np.ndarray, cosmology=DEFAULT_COSMOLOGY) -> np.ndarray:
    distance_mpc = np.asarray(distance_mpc, dtype=float)
    return np.asarray([z_at_value(cosmology.luminosity_distance, d * u.Mpc) for d in distance_mpc], dtype=float)


def differential_comoving_volume_full_sky(z: np.ndarray, cosmology=DEFAULT_COSMOLOGY) -> np.ndarray:
    """dVc/dz over the full sky in Gpc^3."""
    return (cosmology.differential_comoving_volume(z) * 4.0 * np.pi * u.sr).to_value(u.Gpc**3)


def power_law_redshift_pdf(z: np.ndarray, lamb: float, cosmology=DEFAULT_COSMOLOGY) -> np.ndarray:
    """Unnormalized PowerLawRedshift source-frame redshift PDF.

    This follows the standard detected-source counting factor for an underlying
    merger-rate evolution R(z) ∝ (1+z)^lamb:

        p(z | lamb) ∝ (1+z)^lamb * dVc/dz / (1+z).

    The final /(1+z) accounts for cosmological time dilation.
    """
    z = np.asarray(z, dtype=float)
    return np.clip((1.0 + z) ** float(lamb) * differential_comoving_volume_full_sky(z, cosmology=cosmology) / (1.0 + z), 0.0, np.inf)
