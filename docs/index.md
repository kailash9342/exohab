---
title: Exoplanet habitability toolkit for Python
description: >-
  Open-source Python library to compute the Earth Similarity Index, habitable
  zones and Monte Carlo uncertainties for exoplanets, using live NASA Exoplanet
  Archive data. Every formula cited.
hide:
  - navigation
  - toc
---

<div class="exohab-hero" markdown>

![exohab logo](assets/logo.svg)

# exohab

<p class="tagline">How Earth-like is that exoplanet? A small, well-tested Python toolkit for answering that question with real data and honest uncertainties.</p>

[Get started](quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/kailash9342/exohab){ .md-button }

</div>

<div class="exohab-badges" markdown>
[![PyPI](https://img.shields.io/pypi/v/exohab.svg)](https://pypi.org/project/exohab/)
[![Python](https://img.shields.io/pypi/pyversions/exohab.svg)](https://pypi.org/project/exohab/)
[![CI](https://github.com/kailash9342/exohab/actions/workflows/ci.yml/badge.svg)](https://github.com/kailash9342/exohab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/kailash9342/exohab/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24269967.svg)](https://doi.org/10.5281/zenodo.24269967)
</div>

```bash
pip install "exohab[all]"
```

<div class="grid cards" markdown>

-   :material-earth:{ .lg .middle } **Published habitability metrics**

    ---

    Earth Similarity Index (4- and 2-parameter), similarity to Mars, Venus and
    Titan, PHI and BCI, all implemented exactly as published.

    [:octicons-arrow-right-24: Metrics](guide/metrics.md)

-   :material-orbit:{ .lg .middle } **Habitable zones**

    ---

    Kopparapu et al. (2014) conservative and optimistic boundaries, as fluxes
    or orbital distances, for stars from 2600 K to 7200 K.

    [:octicons-arrow-right-24: Habitable zone](guide/habitable-zone.md)

-   :material-dice-multiple:{ .lg .middle } **Uncertainty, not just numbers**

    ---

    Monte Carlo propagation of asymmetric error bars, for one planet or the
    whole catalogue, with 68% and 95% intervals.

    [:octicons-arrow-right-24: Uncertainty](guide/uncertainty.md)

-   :material-satellite-variant:{ .lg .middle } **Live NASA data**

    ---

    Every confirmed planet from the NASA Exoplanet Archive, with no API key and
    a local cache for offline use.

    [:octicons-arrow-right-24: Data](guide/data.md)

-   :material-robot-outline:{ .lg .middle } **ML-ready features**

    ---

    A scikit-learn transformer that turns raw catalogue columns into 15
    habitability features.

    [:octicons-arrow-right-24: Machine learning](guide/ml.md)

-   :material-book-open-variant:{ .lg .middle } **Cited and tested**

    ---

    Every formula is referenced in its docstring, and the test suite checks
    results against values published in the literature.

    [:octicons-arrow-right-24: Science & methods](science.md)

</div>

## Try it in 30 seconds

```python
import exohab as eh

eh.compute_esi_2p(radius=1.12, flux=0.7005)          # Proxima b -> 0.869

mars = eh.compute_esi_4p(radius=0.53, density=0.71, escape_velocity=0.45, surface_temp=240.0)
print(mars.interior, mars.surface, mars.esi)          # ~0.82, ~0.65, ~0.73

mc = eh.compute_esi_2p_mc(radius=(1.34, 0.11, -0.18), flux=(0.66, 0.23, -0.41),
                          n_samples=20_000, seed=42, truncate=True)
print(mc.median, mc.ci68)                             # Kepler-442 b with its error bars
```

![Confirmed exoplanets against the Kopparapu habitable zone, coloured by Earth Similarity Index](assets/img/habitable_zone.png){ loading=lazy }

## Who is it for?

| You are… | exohab gives you… |
|---|---|
| **A student or educator** | Quick, visual comparisons of real worlds with Earth |
| **A data scientist** | Clean DataFrames and engineered habitability features in one call |
| **A researcher** | Cited formulas, custom references and weights, and uncertainty-aware results |

!!! note "Similarity is not habitability"
    The Earth Similarity Index measures how similar a planet's bulk properties
    and insolation are to Earth's. It is **not** a probability of life. See
    [Science & methods](science.md) for the definitions and limitations.
