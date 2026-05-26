#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash validation/run_all.sh /path/to/popsummary.h5 [outdir]"
  exit 2
fi

H5="$1"
OUTDIR="${2:-validation_outputs}"

python validation/00_check_environment.py --h5 "$H5"
python validation/01_inspect_release.py --h5 "$H5" --outdir "$OUTDIR"
python validation/02_sample_grid_marginals.py --h5 "$H5" --outdir "$OUTDIR" --n-hyperrows 1000 --events-per-row 1000 --seed 1234
python validation/03_compare_grid_marginals.py --h5 "$H5" --outdir "$OUTDIR"
python validation/04_check_analytic_sampler_readiness.py --h5 "$H5" --outdir "$OUTDIR"
