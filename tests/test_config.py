from __future__ import annotations

from popsampler.config import read_sampler_config


def test_read_notch_ini_config(tmp_path):
    cfg = tmp_path / "notch.ini"
    cfg.write_text(
        """
[run]
popsummary_file = mock.h5
output = out.parquet
n_events = 123
seed = 7
batch_size = 2
no_extrinsics = true

[model]
name = notch

[grid]
m1_min = 1.0
m1_max = 100.0
mass_grid_size = 333
q_grid_size = 111

[redshift]
model = madau_dickinson
madau_alpha = 2.7
madau_beta = 2.9
madau_z_peak = 1.9

[redshift_evolution]
enabled = true
reference_z = 0.5
alpha_dip = linear_log1pz, gamma_alpha_dip
xi_spin = linear_z, gamma_xi_spin, 0, 1
""".strip()
    )
    out = read_sampler_config(cfg)
    assert out.model == "notch"
    assert out.popsummary_file == "mock.h5"
    assert out.output == "out.parquet"
    assert out.n_events == 123
    assert out.include_extrinsics is False
    assert out.model_config.m1_max == 100.0
    assert out.model_config.mass_grid_size == 333
    assert out.model_config.redshift_rate.model == "madau_dickinson"
    assert out.model_config.redshift_rate.madau_alpha == 2.7
    assert out.model_config.redshift_rate.madau_beta == 2.9
    assert out.model_config.redshift_rate.madau_z_peak == 1.9
    assert out.model_config.redshift_evolution.active
    assert len(out.model_config.redshift_evolution.parameter_evolutions) == 2
