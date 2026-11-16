# exohab

[![CI](https://github.com/kailash9342/exohab/actions/workflows/ci.yml/badge.svg)](https://github.com/kailash9342/exohab/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/exohab.svg)](https://pypi.org/project/exohab/)
[![Python](https://img.shields.io/pypi/pyversions/exohab.svg)](https://pypi.org/project/exohab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/kailash9342/exohab/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24269967.svg)](https://doi.org/10.5281/zenodo.24269967)

**How Earth-like is that exoplanet?** `exohab` is a small, well-tested Python
toolkit for exploring that question with real data. It can:

- 🌍 compute published habitability metrics: the Earth Similarity Index (ESI),
  similarity to Mars, Venus or Titan, PHI, BCI, and Kopparapu habitable-zone limits;
- 🛰️ load live data from the NASA Exoplanet Archive, with no API key and with offline caching;
- 🎲 turn measurement error bars into probability distributions with Monte Carlo;
- 🤖 build scikit-learn-ready features; and
- 📊 plot results in one line.

Every formula comes from a peer-reviewed paper or an official data source and
is cited in the docstring of the function that implements it. Where the
literature has several variants, the exact one used is written down. Nothing
is invented.

## Installation

```bash
pip install exohab
```

Plots and machine-learning helpers need two optional extras (matplotlib and scikit-learn):

```bash
pip install "exohab[all]"
```

`exohab` needs Python 3.10 or newer. The core install depends only on `numpy`,
`pandas` and `requests`.

## Quick start

### 🎓 For students: compare worlds to Earth

```python
import exohab as eh

# Two-parameter ESI from radius (Earth radii) and stellar flux (Earth = 1)
print(eh.compute_esi_2p(radius=1.0, flux=1.0))          # Earth      -> 1.0
print(eh.compute_esi_2p(radius=0.532, flux=0.431))      # Mars       -> ~0.65
print(eh.compute_esi_2p(radius=1.12, flux=0.7005))      # Proxima b  -> ~0.87

# Full four-parameter ESI, split into interior and surface parts (Mars)
res = eh.compute_esi_4p(radius=0.53, density=0.71, escape_velocity=0.45, surface_temp=240.0)
print(res.interior, res.surface, res.esi)                # ~0.82, ~0.65, ~0.73

# How Mars-like is Earth? (Mars Similarity Index)
print(eh.compute_similarity(1.0, 1.0, 1.0, 288.0, reference="mars").esi)   # ~0.68

# Every confirmed planet, plotted against the habitable zone
df = eh.load_nasa_exoplanets()
eh.plot_habitable_zone(df, color_col="esi_2p", labels=["Kepler-442 b", "TOI-700 d"])
```

### 🔬 For researchers: propagate asymmetric uncertainties

```python
import exohab as eh

# Kepler-442 b: R = 1.34 +0.11/-0.18 R_E, S = 0.66 +0.23/-0.41 S_E (NASA Exoplanet Archive)
mc = eh.compute_esi_2p_mc(radius=(1.34, 0.11, -0.18), flux=(0.66, 0.23, -0.41),
                          n_samples=20_000, seed=42, truncate=True)
print(mc.median, mc.ci68, mc.ci95)

# TRAPPIST-1 e, 4-parameter ESI. Density and escape velocity are derived per
# draw from mass and radius, so their correlation is kept. The equilibrium
# temperature (249.7 +/- 2.4 K) is converted to a surface-temperature estimate.
ts = eh.estimate_surface_temp(249.7)                     # ~283.3 K
mc4 = eh.compute_esi_mc(radius=(0.920, 0.013, -0.012), mass=(0.692, 0.022, -0.022),
                        surface_temp=(ts, 1.096 * 2.4), n_samples=20_000, seed=42)
print(mc4.median, mc4.components["interior"].median, mc4.components["surface"].median)
eh.plot_mc_distribution(mc4, label="ESI (TRAPPIST-1 e)")

# Uncertainty for the whole catalogue, from the archive's own err1/err2 columns
df = eh.load_nasa_exoplanets(surface_temp_method="kashyap2017")
unc = eh.compute_esi_mc_catalog(df, method="2p", n_samples=2000, seed=0)
df.join(unc).nlargest(10, "esi_2p_mc_median")[
    ["pl_name", "esi_2p", "esi_2p_mc_median", "esi_2p_mc_ci68_low", "esi_2p_mc_ci68_high"]
]
```

### 🤖 For ML engineers: habitability features in a scikit-learn pipeline

```python
import exohab as eh
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

df = eh.load_nasa_exoplanets()

features = eh.HabitabilityFeatureTransformer(surface_temp_method="kashyap2017")
X = features.fit_transform(df)            # DataFrame with 15 engineered features
print(list(features.get_feature_names_out()))

model = make_pipeline(
    eh.HabitabilityFeatureTransformer(surface_temp_method="kashyap2017"),
    SimpleImputer(strategy="median"),
    KMeans(n_clusters=4, n_init="auto", random_state=0),
)
df["cluster"] = model.fit_predict(df)
```

More examples:

- [`examples/quickstart.ipynb`](https://github.com/kailash9342/exohab/blob/main/examples/quickstart.ipynb) is a runnable notebook walkthrough.
- [`examples/full_tour.py`](https://github.com/kailash9342/exohab/blob/main/examples/full_tour.py) is one script that runs and checks every public function.
- [`examples/try_exohab.py`](https://github.com/kailash9342/exohab/blob/main/examples/try_exohab.py) is a short check of your installation against published values.

## What is included

| Area | Functions | Definition used |
|---|---|---|
| Earth Similarity Index | `compute_esi_4p` | Schulze-Makuch et al. (2011): `ESI_x = (1 - abs(x-x0)/(x+x0))**w`; weights are radius 0.57, density 1.07, escape velocity 0.70 and surface temperature 5.58; interior, surface and global indices are geometric means |
| Two-parameter ESI | `compute_esi_2p` | Planetary Habitability Laboratory: `1 - sqrt(0.5*[((S-1)/(S+1))**2 + ((R-1)/(R+1))**2])` |
| Similarity to other bodies | `compute_similarity` | Same form with another body's reference values. `"mars"` is the published Mars Similarity Index (Kashyap Jagadeesh et al. 2017). `"venus"` and `"titan"` use NASA fact-sheet values with the Earth ESI weights, because no body-specific weights have been published. Custom dicts also work. |
| Planetary Habitability Index | `compute_phi` | `(S*E*C*L)**(1/4)` (Schulze-Makuch et al. 2011); you supply the factor scores in [0, 1] |
| Biological Complexity Index | `compute_bci` | `(S*E*T*G*A)**(1/5)`, optional `BCI_rel = BCI/max` (Irwin et al. 2014); you supply the factor scores in [0, 1] |
| Surface temperature | `estimate_surface_temp` | `T_s = 9.650 + 1.096*T_eq` (Kashyap Jagadeesh 2017); rocky planets with atmospheres only |
| Habitable zone | `kopparapu_seff`, `hz_distance` | Kopparapu et al. (2014) flux limits: recent Venus, runaway greenhouse (0.1/1/5 Earth masses), maximum greenhouse and early Mars; valid for 2600–7200 K; `d = sqrt(L/S_eff)` |
| Uncertainty | `compute_esi_mc`, `compute_esi_2p_mc`, `compute_esi_mc_catalog`, `sample_split_normal`, `split_normal_cdf` | Two-piece (split) normal for quotes like `x +a -b`, with optional truncation at zero |
| Data | `load_nasa_exoplanets`, `add_habitability_columns` | NASA Exoplanet Archive TAP service, `pscomppars` table |
| Machine learning | `HabitabilityFeatureTransformer` | Derived density and escape velocity, ESI scores, interior/surface ratio, HZ fluxes and distances, and in-HZ flags |
| Plots | `plot_esi_scatter`, `plot_habitable_zone`, `plot_mc_distribution` | matplotlib |

All metric functions accept scalars, lists, NumPy arrays and pandas Series.
Series indices are kept. Missing or invalid values give NaN instead of an
error, so you can process a whole catalogue in one call.

## Scientific notes and limitations

Please read these before using the results in research:

- **Surface vs. equilibrium temperature.** The four-parameter ESI is defined
  with the mean *surface* temperature (Earth: 288 K). Catalogues list the
  *equilibrium* temperature instead (Earth: about 255 K).
  - By default `pl_eqt` is used as-is, which lowers the surface ESI of temperate planets.
  - `surface_temp_method="kashyap2017"` applies an empirical fit to five Solar System bodies instead.
  - That fit is not valid for gas giants, and it extrapolates above about 255 K.
- **Mass.** `pl_bmasse` is the archive's "best mass". For radial-velocity
  planets this may be a minimum mass (M sin i).
- **ESI measures similarity, not the probability of life.** A high ESI means
  Earth-like bulk properties and insolation, nothing more.
- **PHI and BCI need expert factor scores.** `exohab` implements only the
  published formulas that combine those scores. It does not invent scores from raw data.
- **Monte Carlo inputs are sampled independently.** The exception is mass
  and radius, whose correlation is kept when you pass `mass`.
  - Missing error bars are treated as zero and flagged in `*_errors_complete`.
  - For those planets the intervals are lower limits on the true uncertainty.
- **Kopparapu habitable-zone limits are NaN outside 2600–7200 K** unless you
  pass `extrapolate=True`.

## Data source and acknowledgement

Live data comes from the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
`load_nasa_exoplanets` caches each query in `~/.exohab/cache`, so later calls
work offline. You can change the location with `cache_dir=` or the
`EXOHAB_CACHE_DIR` environment variable, and force a fresh download with
`refresh=True`.

If you publish results based on this data, the archive asks you to include
this acknowledgement:

> This research has made use of the NASA Exoplanet Archive, which is operated
> by the California Institute of Technology, under contract with the National
> Aeronautics and Space Administration under the Exoplanet Exploration Program.

**Behind a corporate proxy or antivirus HTTPS scanning?** If you see
`SSL: CERTIFICATE_VERIFY_FAILED`, install `truststore` with
`pip install truststore`. Then run
`import truststore; truststore.inject_into_ssl()` before loading data. This
uses your operating system's certificate store. Please do not disable
certificate verification.

## References

- Schulze-Makuch, D. et al. (2011). A Two-Tiered Approach to Assessing the Habitability of Exoplanets. *Astrobiology* 11(10), 1041–1052. doi:10.1089/ast.2010.0592
- Planetary Habitability Laboratory, UPR Arecibo. Earth Similarity Index. https://phl.upr.edu/projects/earth-similarity-index-esi
- Kashyap Jagadeesh, M. (2017). *Earth Similarity Index and Habitability Studies of Exoplanets.* M.Phil. dissertation, Christ University. arXiv:1801.07101
- Irwin, L. N., Méndez, A., Fairén, A. G., Schulze-Makuch, D. (2014). Assessing the Possibility of Biological Complexity on Other Worlds. *Challenges* 5(1), 159–174. doi:10.3390/challe5010159
- Kopparapu, R. K. et al. (2013). Habitable Zones around Main-sequence Stars: New Estimates. *ApJ* 765, 131. arXiv:1301.6674
- Kopparapu, R. K. et al. (2014). Habitable Zones around Main-sequence Stars: Dependence on Planetary Mass. *ApJL* 787, L29. arXiv:1404.5292
- Wallis, K. F. (2014). The Two-Piece Normal, Binormal, or Double Gaussian Distribution. *Statistical Science* 29(1), 106–112.

## Citing exohab

If `exohab` helps your work, please cite it:

```bibtex
@software{sivagnanam_exohab,
  author  = {Sivagnanam, Kailash},
  title   = {exohab: exoplanet habitability toolkit for Python},
  version = {0.2.0},
  year    = {2026},
  doi     = {10.5281/zenodo.24269968},
  url     = {https://doi.org/10.5281/zenodo.24269968},
  license = {MIT}
}
```

Each release is archived on Zenodo with its own DOI. The DOI above is for
version 0.2.0. To cite the project as a whole, which always resolves to the
latest version, use
[10.5281/zenodo.24269967](https://doi.org/10.5281/zenodo.24269967).

The same metadata is in [`CITATION.cff`](https://github.com/kailash9342/exohab/blob/main/CITATION.cff). GitHub reads that file
and shows a "Cite this repository" button. Please also cite the original
papers of the metrics you use, listed above.

## Contributing and development

```bash
git clone https://github.com/kailash9342/exohab.git
cd exohab
pip install -e ".[dev,all]"
pytest
```

All tests run offline. Network calls to the NASA archive are mocked. Bug
reports and suggestions are welcome on the
[issue tracker](https://github.com/kailash9342/exohab/issues).

## License

MIT. See [LICENSE](https://github.com/kailash9342/exohab/blob/main/LICENSE).
