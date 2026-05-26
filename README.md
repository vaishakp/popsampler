# popsampler

Posterior-predictive compact-binary population sampling utilities, initially targeted at the GWTC-4.0 BBH population release.

This repository is being built in staged form:

1. Download and cache LVK GWTC-4.0 population `popsummary` products.
2. Inspect the released hyperposterior samples and model metadata.
3. Implement model-specific posterior-predictive source sampling without replacing joint distributions by independent marginals.

The first implementation stage intentionally prioritizes reproducibility and introspection over guessing model parameter mappings.
