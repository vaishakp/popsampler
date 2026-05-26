# Validation workflow

These scripts validate the current GWTC-4 `popsummary` interface in small, reproducible steps. They are meant to be run from the repository root after installing the package in editable mode.

```bash
python -m pip install -e '.[gwtc4,parquet,dev]'
```

Set the HDF5 file once:

```bash
export H5=data/gwtc4/analyses_BBH/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5
```

Then run the scripts one by one:

```bash
python validation/00_check_environment.py --h5 "$H5"
python validation/01_inspect_release.py --h5 "$H5" --outdir validation_outputs
python validation/02_sample_grid_marginals.py --h5 "$H5" --outdir validation_outputs --n-hyperrows 1000 --events-per-row 1000 --seed 1234
python validation/03_compare_grid_marginals.py --h5 "$H5" --outdir validation_outputs
python validation/04_check_analytic_sampler_readiness.py --h5 "$H5" --outdir validation_outputs
```

Or run everything:

```bash
bash validation/run_all.sh "$H5"
```

## What each step checks

1. `00_check_environment.py`
   - verifies imports;
   - verifies that the HDF5 file exists;
   - prints package/library versions.

2. `01_inspect_release.py`
   - lists the HDF5 tree;
   - lists available `posterior/rates_on_grids` products;
   - loads the hyperparameter sample table through `popsummary`;
   - writes `validation_outputs/hdf5_tree.txt`, `rate_grids.csv`, and `hyperparameter_columns.txt`.

3. `02_sample_grid_marginals.py`
   - chooses hyperposterior rows;
   - samples from the released 1D marginal rate grids row by row;
   - writes `grid_marginal_samples.parquet`, `grid_rows.npy`, and `grid_marginal_summary.csv`.

4. `03_compare_grid_marginals.py`
   - compares the Monte Carlo samples from step 2 against the released grid-averaged rate curves;
   - writes `grid_marginal_diagnostics.csv`.

5. `04_check_analytic_sampler_readiness.py`
   - checks whether the hyperparameter table has named columns;
   - imports the analytic sampler;
   - attempts name-based hyperparameter validation only if named columns are available.

## Interpretation

The grid-marginal workflow validates that we can read and sample the released 1D posterior rate grids. This is a validation/debugging path, not the final correlated CE population generator. The released grids reproduce 1D marginals only; they do not by themselves encode joint correlations such as `mass_1-mass_ratio` or `q-chi_eff`.
