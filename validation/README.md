# Validation workflow

These scripts validate the current GWTC-4 `popsummary` interface in small, reproducible steps. They are meant to be run from the repository root after installing the package in editable mode.

```bash
python -m pip install -e '.[gwtc4,parquet,plot,dev]'
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
python validation/08_validate_joint_sampler_against_grids.py --h5 "$H5" --outdir validation_outputs --n-hyperrows 50 --events-per-row 2000 --seed 1234 --no-extrinsics
python validation/13_plot_1d_model_comparisons.py --h5 "$H5" --outdir validation_outputs/figures/sample_1d
python validation/14_validate_analytic_1d_marginals.py --h5 "$H5" --outdir validation_outputs --n-hyperrows 50 --seed 1234
python validation/15_plot_analytic_1d_marginals.py --outdir validation_outputs/figures/analytic_1d
python validation/16_plot_2d_projections.py --h5 "$H5" --outdir validation_outputs/figures/2d
```

Or run the older core grid workflow:

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

6. `08_validate_joint_sampler_against_grids.py`
   - draws posterior-predictive samples from the named analytic sampler;
   - compares sampled 1D projections for masses/spins against released 1D grids;
   - writes `joint_sampler_validation_samples.parquet` and `joint_sampler_grid_diagnostics.csv`.

7. `13_plot_1d_model_comparisons.py`
   - plots sampled 1D histograms against released 1D grids;
   - separately plots the redshift rate-density check.

8. `14_validate_analytic_1d_marginals.py`
   - evaluates analytic 1D model marginals row by row;
   - compares them directly against released 1D grid curves without Monte Carlo noise;
   - writes `analytic_1d_marginal_diagnostics.csv` and per-parameter curve CSVs.

9. `15_plot_analytic_1d_marginals.py`
   - makes one plot per 1D parameter: analytic model curve vs released grid curve.

10. `16_plot_2d_projections.py`
   - computes and plots analytic equal-row `p(mass_1, mass_ratio)` contours;
   - if `joint_sampler_validation_samples.parquet` exists, overlays sampled `mass_1-mass_ratio` projections and plots additional sampled projections such as `mass_1-mass_2`, `mass_1-chi_eff`, and `q-chi_eff`.

## Interpretation

The grid-marginal workflow validates that we can read and sample the released 1D posterior rate grids. This is a validation/debugging path, not the final correlated CE population generator. The released grids reproduce 1D marginals only; they do not by themselves encode joint correlations such as `mass_1-mass_ratio` or `q-chi_eff`.

The analytic 1D workflow is stricter: it evaluates the GWTC-4 model formulae implied by selected hyperposterior rows and compares those curves to the released 1D grids. This is the best automated check for formula and convention errors.

The 2D workflow is a model/sample self-consistency diagnostic unless the HDF5 file exposes explicit multidimensional rate grids. For the current default release products, the robust 2D reference we can construct is the model-implied analytic `p(mass_1, mass_ratio)`; other 2D planes involving spins or `chi_eff` are sampled posterior-predictive projections rather than direct comparisons to a released 2D grid.
