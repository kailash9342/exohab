---
description: Complete API reference for exohab - every function, class and parameter, generated from the source code docstrings.
---

# API reference

Everything listed here is available directly from the top-level package, for
example `exohab.compute_esi_4p`. This page is generated from the docstrings in
the source code.

## Metrics

::: exohab.metrics
    options:
      show_root_heading: false
      members:
        - compute_esi_4p
        - compute_esi_2p
        - compute_similarity
        - compute_phi
        - compute_bci
        - density_from_mass_radius
        - escape_velocity_from_mass_radius
        - estimate_surface_temp
        - ESIResult
        - ESI_WEIGHTS
        - REFERENCE_BODIES
        - SURFACE_TEMP_METHODS

## Habitable zone

::: exohab.habitable_zone
    options:
      show_root_heading: false
      members:
        - kopparapu_seff
        - hz_distance
        - KOPPARAPU_2014
        - TEFF_RANGE

## Uncertainty

::: exohab.uncertainty
    options:
      show_root_heading: false
      members:
        - compute_esi_mc
        - compute_esi_2p_mc
        - compute_esi_mc_catalog
        - MCResult
        - sample_split_normal
        - split_normal_cdf
        - parse_measurement

## Data

::: exohab.data
    options:
      show_root_heading: false
      members:
        - load_nasa_exoplanets
        - add_habitability_columns
        - build_query
        - default_cache_dir
        - DEFAULT_COLUMNS
        - TAP_URL

## Machine learning

::: exohab.ml
    options:
      show_root_heading: false
      members:
        - HabitabilityFeatureTransformer
        - FEATURE_NAMES

## Plots

::: exohab.viz
    options:
      show_root_heading: false
      members:
        - plot_esi_scatter
        - plot_habitable_zone
        - plot_mc_distribution
