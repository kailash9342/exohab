# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-11-16

No changes to the library code or results. This release updates the project
metadata and documentation shown on PyPI.

### Added
- Documentation website at https://kailash9342.github.io/exohab/ (MkDocs
  Material on GitHub Pages): user guide, science & methods with formulas and
  validation, an API reference generated from docstrings, examples and
  citation details. It's rebuilt and deployed automatically on every push to
  `main`.
- Zenodo DOI ([10.5281/zenodo.24269967](https://doi.org/10.5281/zenodo.24269967))
  in the README and `CITATION.cff`.

### Changed
- PyPI "Homepage" and "Documentation" links point to the documentation
  website.
- README: documentation link, DOI badge, and a citation that uses the
  all-versions DOI.
- Release workflow: refuses to publish if the tag doesn't match the package
  version; publishes to PyPI through trusted publishing (no API tokens).

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
- README with quick starts for students, researchers and ML engineers, plus
  formulas, limitations and references; `CITATION.cff`.
- `examples/quickstart.ipynb`: an executed walkthrough with live NASA data,
  Monte Carlo and plots.
- GitHub Actions: `ci.yml` (pytest on Ubuntu with Python 3.10–3.12, a
  core-only install, and a build with `twine check`) and `publish.yml` (PyPI
  trusted publishing).
- `tests/test_docs.py`: checks that every `exohab` call in the README and the
  notebook uses functions and parameters that exist.

### Packaging
- License metadata now uses the PEP 639 form: `license = "MIT"` with
  `license-files`.

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
