---
title: About exohab and its author
description: >-
  About exohab, the open-source Python exoplanet habitability toolkit, and its
  creator Kailash Sivagnanam, Founder & CEO of Hacroo Technologies and AI &
  Data Science graduate from Panruti, Tamil Nadu: mission, principles and
  acknowledgements.
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

![Kailash Sivagnanam, founder and CEO of Hacroo Technologies and creator of exohab](assets/img/kailash-sivagnanam.jpg){ width="132" height="132" }

<div markdown>

### Kailash Sivagnanam

<p class="hx-role">Founder &amp; CEO, Hacroo Technologies · Creator of exohab</p>

:material-school: Graduate in Artificial Intelligence & Data Science &nbsp;·&nbsp; :material-map-marker: Panruti, Tamil Nadu, India
{ .hx-meta }

Kailash Sivagnanam is the founder and CEO of
[Hacroo Technologies](https://hacroo.com/) and a graduate in Artificial
Intelligence & Data Science, with a passion for astronomy and space
technology. Kailash created exohab to bring those interests together: rigorous,
openly cited science and modern data tools that make exoplanet habitability
research accessible to students, data scientists and researchers. Kailash
designs and maintains every part of the project, from the habitability
metrics and uncertainty modelling to the NASA data pipeline, the tests and
this website.

<div class="hx-author-links" markdown>
[:fontawesome-brands-linkedin: LinkedIn](https://www.linkedin.com/in/kailash-sivaganam-49a383315){ .md-button }
[:fontawesome-brands-github: GitHub](https://github.com/kailash9342){ .md-button }
[:material-web: hacroo.com](https://hacroo.com/){ .md-button }
[:fontawesome-brands-python: PyPI](https://pypi.org/user/kailash-sivagnanam/){ .md-button }
</div>

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
