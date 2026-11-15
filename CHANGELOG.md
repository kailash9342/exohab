# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-15

### Added
- `estimate_surface_temp`: surface temperature from equilibrium temperature,
  `T_s = 9.650 + 1.096 T_eq` (Kashyap Jagadeesh 2017). Available in
  `load_nasa_exoplanets`, `add_habitability_columns` and
  `HabitabilityFeatureTransformer` through `surface_temp_method="kashyap2017"`.
  The temperature used for the 4-parameter ESI is recorded in
  `DataFrame.attrs["esi_4p_temperature"]`.
- `compute_esi_mc_catalog`: vectorised Monte Carlo ESI summaries for every
  planet, using the archive's `err1`/`err2` columns. Rows with missing errors
  are flagged.
- `truncate=True` option in the Monte Carlo functions: non-positive draws are
  redrawn (split normal truncated at zero), so the full sample size is kept.
- `split_normal_cdf` and a `lower` bound argument for `sample_split_normal`.
- `plot_habitable_zone`: effective temperature vs. insolation diagram with the
  Kopparapu et al. (2014) conservative and optimistic habitable zones.
- README with formulas, limitations and references; `CITATION.cff`;
  GitHub Actions workflows for tests and PyPI publishing.

### Changed
- `plot_esi_scatter` now stacks the candidate labels in a column with leader
  lines, so labels no longer overlap when candidates cluster near Earth.

### Removed
- `scipy` is no longer a dependency; it was never used. The core install is
  now `numpy`, `pandas` and `requests`.

## [0.1.0] - 2026-11-15

### Added
- Initial release: 4- and 2-parameter ESI, reference-body similarity (Earth,
  Mars MSI, Venus, Titan), PHI, BCI, Kopparapu (2014) habitable-zone limits,
  split-normal Monte Carlo uncertainty, NASA Exoplanet Archive loader with
  local cache, scikit-learn feature transformer and plots.
