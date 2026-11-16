---
title: About exohab and its author
description: >-
  About exohab, the open-source Python exoplanet habitability toolkit, and its
  creator and maintainer Kailash Sivagnanam: mission, principles,
  acknowledgements and how to contribute.
---

# About exohab

**exohab** is a free, open-source Python toolkit for exploring a simple
question with real data: *how Earth-like is a given exoplanet?*

## Why it exists

More than six thousand exoplanets have been confirmed, but answering even a
basic habitability question usually means repeating the same work: copying
formulas from papers, downloading catalogue tables by hand and ignoring the
measurement error bars. exohab packages that work once, carefully, so
students, data scientists and researchers can focus on the questions instead
of the boilerplate.

## Principles

<div class="grid cards" markdown>

-   :material-book-check:{ .lg .middle } **Cite everything**

    ---

    Every formula comes from a peer-reviewed paper or an official data source
    and is cited in the docstring that implements it. Where the literature has
    variants, the exact one used is written down.

-   :material-test-tube:{ .lg .middle } **Test against the literature**

    ---

    The automated test suite checks results against values published in the
    papers, and it runs on every change.

-   :material-alert-circle-check-outline:{ .lg .middle } **Be honest about limits**

    ---

    ESI measures similarity, not the probability of life. Limitations such as
    equilibrium vs. surface temperature are documented, not hidden.

-   :material-feather:{ .lg .middle } **Stay lightweight and open**

    ---

    The core depends only on numpy, pandas and requests. The code is MIT
    licensed and archived with a DOI.

</div>

## Author

<div class="hx-author" markdown>

![Kailash Sivagnanam](https://github.com/kailash9342.png?size=216){ width="108" height="108" }

<div markdown>

### Kailash Sivagnanam

<p class="hx-role">Creator and maintainer of exohab</p>

Kailash designed and maintains exohab, from the habitability metrics and the
uncertainty modelling to the NASA data pipeline, the tests and this website.

[:fontawesome-brands-github: GitHub](https://github.com/kailash9342){ .md-button }
[:fontawesome-brands-python: PyPI](https://pypi.org/user/kailash-sivagnanam/){ .md-button }

</div>
</div>

## Project facts

| Fact | Details |
|---|---|
| **First public release** | 0.2.0 (November 2026) |
| **License** | MIT |
| **Language** | Python 3.10+ |
| **Install** | `pip install exohab` |
| **Source code** | [github.com/kailash9342/exohab](https://github.com/kailash9342/exohab) |
| **Package** | [pypi.org/project/exohab](https://pypi.org/project/exohab/) |
| **DOI** | [10.5281/zenodo.24269967](https://doi.org/10.5281/zenodo.24269967) |

## Acknowledgements

exohab stands on the work of others:

- The **NASA Exoplanet Archive**, the source of the planetary data. *This
  research has made use of the NASA Exoplanet Archive, which is operated by the
  California Institute of Technology, under contract with the National
  Aeronautics and Space Administration under the Exoplanet Exploration
  Program.*
- The authors of the methods exohab implements: Schulze-Makuch et al.,
  Kopparapu et al., Irwin et al., Kashyap Jagadeesh et al. and the
  **Planetary Habitability Laboratory** (see [Science & methods](science.md#references)).
- The open-source scientific Python ecosystem: NumPy, pandas, matplotlib,
  scikit-learn, and Material for MkDocs for this website.

exohab is an independent project. It is not affiliated with or endorsed by
NASA or the Planetary Habitability Laboratory.

## Contribute and get in touch

- **Found a bug or have an idea?** Open an
  [issue on GitHub](https://github.com/kailash9342/exohab/issues).
- **Want to contribute code or documentation?** Pull requests are welcome; see
  the [development install](installation.md#development-install).
- **Using exohab in your work?** Please [cite it](citing.md). It helps the
  project grow.
