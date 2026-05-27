from __future__ import annotations

import numpy as np

from popsampler.model_registry import get_bbh_model
from popsampler.models import BBHDefaultModelConfig
from popsampler.notch_mass import NotchMassSampler, NotchMassSamplerConfig


ROW = {
    "alpha_1": -4.5,
    "alpha_dip": -1.7,
    "alpha_2": -0.9,
    "NSmin": 1.1,
    "NSmax": 3.0,
    "BHmin": 6.0,
    "BHmax": 80.0,
    "beta": 1.2,
    "mu_chi": 0.25,
    "sigma_chi": 0.2,
    "mu_spin": 0.0,
    "sigma_spin": 0.8,
    "xi_spin": 0.4,
    "lamb": 2.0,
}


def test_notch_mass_sampler_bounds_and_columns():
    sampler = NotchMassSampler(NotchMassSamplerConfig(mass_grid_size=256, q_grid_size=128))
    out = sampler.sample(ROW, 200, rng=np.random.default_rng(1234))
    assert "mass_1_source" in out
    assert "mass_2_source" in out
    assert "mass_ratio" in out
    assert "chirp_mass_source" in out
    assert np.all(out["mass_1_source"] >= out["mass_2_source"])
    assert np.all(out["mass_2_source"] >= ROW["NSmin"])
    assert np.all(out["mass_1_source"] <= ROW["BHmax"])
    assert np.all((out["mass_ratio"] >= 0.001) & (out["mass_ratio"] <= 1.0))


def test_registry_instantiates_notch_model_and_draws_samples():
    config = BBHDefaultModelConfig(
        mass_grid_size=128,
        q_grid_size=64,
        z_grid_size=128,
        spin_grid_size=128,
        warn_if_unvalidated=False,
    )
    model = get_bbh_model("notch", config)
    samples = model.sample(ROW, 20, rng=np.random.default_rng(5678), include_extrinsics=False)
    assert len(samples) == 20
    assert samples["sampler_mode"].iloc[0] == "native_notch_joint_model"
