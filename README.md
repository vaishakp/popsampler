# popsampler

Posterior-predictive compact-binary population sampling utilities, initially targeted at the GWTC-4.0 BBH population release.

## Design principle

The scientifically correct posterior-predictive workflow is

```text
Λ_i ~ p(Λ | GWTC-4 population data)
θ_i ~ p(θ | Λ_i)
```

This repository therefore avoids reconstructing joint event distributions from independent 1D marginals. The first implementation stage adds downloader/inspection tooling plus conservative BBH model samplers. The sampler validates hyperparameter names from the downloaded `popsummary` file before drawing samples, because the exact released schema is the source of truth.

## Current stage

Implemented in the `initial-gwtc4-sampler` branch:

1. `popsampler-download-gwtc4`: download/cache/extract the GWTC-4.0 `analyses_BBH.tar` release product.
2. `popsampler-inspect`: inspect an HDF5 `popsummary` file and print available hyperparameter columns.
3. `popsampler-sample-bbh`: draw source samples by drawing coherent hyperposterior rows and sampling a model conditional on each row.
4. User-selectable mass models: `bpl2peak` and native `notch`.
5. Optional redshift evolution: draw `z` first, evolve selected hyperparameters to `Λ(z)`, then draw masses/spins conditional on the evolved row.

The default sampler remains the GWTC-style block-factorized `bpl2peak` model. Redshift evolution and the Notch model are opt-in and must be explicitly requested.

## Install

```bash
python -m pip install -e '.[gwtc4,parquet]'
```

## Example

```bash
popsampler-download-gwtc4 --cache-dir data/gwtc4

popsampler-inspect \
  data/gwtc4/analyses_BBH/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5

popsampler-sample-bbh \
  data/gwtc4/analyses_BBH/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5 \
  --n-events 10000 \
  --seed 1234 \
  --output samples.parquet
```

## User-selectable mass models

The sampler now accepts a user-facing model choice:

```bash
popsampler-sample-bbh "$H5" \
  --model bpl2peak \
  --n-events 10000 \
  --output bpl2peak_samples.parquet

popsampler-sample-bbh "$H5" \
  --model notch \
  --n-events 10000 \
  --output notch_samples.parquet
```

The native Notch mass model is implemented inside `popsampler`; it does not call GWForge, gwpopulation, or another population-model package.  Its object spectrum is a continuous three-segment power law over `[NSmin, NSmax]`, `[NSmax, BHmin]`, and `[BHmin, BHmax]`, with slopes `alpha_1`, `alpha_dip`, and `alpha_2`. Binary masses are drawn from

```text
p(m1, q | Λ) ∝ p_obj(m1 | Λ) p_obj(q m1 | Λ) q**beta
```

with ordered components `m1 >= m2`, `q = m2/m1`, and the same object-spectrum support applied to both components. `BHmax`/`bh_max`/`mmax` and `beta` are optional; if absent, `BHmax` is taken from the grid `m1_max`, and `beta = 0`.

## INI-style configuration

Instead of passing all options on the command line, create an INI file:

```ini
[run]
popsummary_file = /path/to/popsummary.h5
output = notch_samples.parquet
n_events = 10000
seed = 1234
batch_size = 1
no_extrinsics = false

[model]
name = notch
warn_if_unvalidated = true

[grid]
m1_min = 1.0
m1_max = 300.0
q_min = 0.001
q_max = 1.0
mass_grid_size = 1200
q_grid_size = 500
z_grid_size = 4096
spin_grid_size = 2048

[redshift_evolution]
enabled = false
reference_z = 0.0
```

Run it with:

```bash
popsampler-sample-bbh --config examples/notch_sampler.ini
```

CLI arguments override the INI values, so this is also valid:

```bash
popsampler-sample-bbh --config examples/notch_sampler.ini \
  --model bpl2peak \
  --output bpl2peak_samples.parquet
```

## Optional redshift evolution

Use `--redshift-evolve` to evolve one or more hyperparameters before drawing event intrinsics. The CLI syntax is:

```text
PARAM:KIND:COEFF[:CLIP_MIN:CLIP_MAX]
```

where `KIND` is one of:

- `power_law`: `theta(z) = theta_ref * [(1 + z)/(1 + z_ref)]**COEFF`
- `linear_z`: `theta(z) = theta_ref + COEFF * (z - z_ref)`
- `linear_log1pz`: `theta(z) = theta_ref + COEFF * log[(1 + z)/(1 + z_ref)]`

`COEFF` can be a literal number or the name of a hyperposterior column. The optional clip bounds are useful for parameters with compact support, such as `xi_spin`.

Examples:

```bash
# Evolve the upper Gaussian mass peak with a fixed power-law exponent.
popsampler-sample-bbh "$H5" \
  --n-events 10000 \
  --output evolved_mpp2.parquet \
  --redshift-evolve mpp_2:power_law:1.0

# Evolve beta additively in log(1+z), and clip xi_spin to [0, 1].
popsampler-sample-bbh "$H5" \
  --n-events 10000 \
  --output evolved_multi.parquet \
  --redshift-evolve beta:linear_log1pz:-0.5 \
  --redshift-evolve xi_spin:linear_z:gamma_xi_spin:0:1
```

INI syntax for the same mechanism is:

```ini
[redshift_evolution]
enabled = true
reference_z = 0.0
alpha_dip = linear_log1pz, gamma_alpha_dip
BHmin = power_law, gamma_BHmin
xi_spin = linear_z, gamma_xi_spin, 0, 1
```

For publication-grade use, the evolution coefficients should correspond to an actual hyperposterior model. Fixed coefficients are intended for controlled injections, stress tests, and sensitivity studies.

## Important caveat

The default BBH and native Notch samplers are initial implementations and should be validated against the LVK figure scripts/rate grids before being used for publication-grade posterior predictive samples. The downloader and inspector are intended to make that validation explicit rather than hiding assumptions.
