# popsampler

Posterior-predictive compact-binary population sampling utilities, initially targeted at the GWTC-4.0 BBH population release.

## Design principle

The scientifically correct posterior-predictive workflow is

```text
Λ_i ~ p(Λ | GWTC-4 population data)
θ_i ~ p(θ | Λ_i)
```

This repository therefore avoids reconstructing joint event distributions from independent 1D marginals. The first implementation stage adds downloader/inspection tooling plus a conservative default BBH model sampler. The sampler validates hyperparameter names from the downloaded `popsummary` file before drawing samples, because the exact released schema is the source of truth.

## Current stage

Implemented in the `initial-gwtc4-sampler` branch:

1. `popsampler-download-gwtc4`: download/cache/extract the GWTC-4.0 `analyses_BBH.tar` release product.
2. `popsampler-inspect`: inspect an HDF5 `popsummary` file and print available hyperparameter columns.
3. `popsampler-sample-bbh`: draw source samples by drawing coherent hyperposterior rows and sampling a model conditional on each row.
4. Optional redshift evolution: draw `z` first, evolve selected hyperparameters to `Λ(z)`, then draw masses/spins conditional on the evolved row.

The default sampler remains the GWTC-style block-factorized model. Redshift evolution is opt-in and must be explicitly requested.

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

For publication-grade use, the evolution coefficients should correspond to an actual hyperposterior model. Fixed coefficients are intended for controlled injections, stress tests, and sensitivity studies.

## Important caveat

The default BBH model sampler is an initial implementation and should be validated against the LVK figure scripts/rate grids before being used for publication-grade posterior predictive samples. The downloader and inspector are intended to make that validation explicit rather than hiding assumptions.
