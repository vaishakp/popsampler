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

## Important caveat

The default BBH model sampler is an initial implementation and should be validated against the LVK figure scripts/rate grids before being used for publication-grade posterior predictive samples. The downloader and inspector are intended to make that validation explicit rather than hiding assumptions.
