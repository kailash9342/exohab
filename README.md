# exohab

**Exoplanet habitability toolkit for Python** — fetch live data from the NASA
Exoplanet Archive, compute published habitability metrics, propagate
measurement uncertainties with Monte Carlo, engineer ML features and plot the
results, in a few lines of code.

Every formula in `exohab` comes from a peer-reviewed paper or an official data
source, and is cited in the docstring of the function that implements it.
Where the literature has several variants, the exact one used is documented.

## Installation

```bash
pip install exohab            # core: numpy, pandas, requests
pip install "exohab[all]"     # + matplotlib (plots) and scikit-learn (ML)
```

Requires Python 3.10 or newer.

## Quick start

```python
import exohab as eh

# Earth Similarity Index from radius and stellar flux (transiting planets)
eh.compute_esi_2p(radius=1.12, flux=0.7005)          # Proxima b -> 0.869

# Full 4-parameter ESI: radius, density, escape velocity (Earth units), surface T (K)
res = eh.compute_esi_4p(0.53, 0.71, 0.45, 240.0)     # Mars
res.interior, res.surface, res.esi                   # -> 0.82, 0.65, 0.73

# Asymmetric error bars -> probability distribution instead of one number
mc = eh.compute_esi_2p_mc(radius=(1.34, 0.11, -0.18), flux=(0.66, 0.23, -0.41), seed=1)
mc.median, mc.ci68, mc.ci95

# Live catalogue (no API key), cached in ~/.exohab/cache for offline reuse
df = eh.load_nasa_exoplanets(surface_temp_method="kashyap2017")
df.nlargest(10, "esi_2p")[["pl_name", "pl_rade", "pl_insol", "esi_2p", "esi_4p"]]

# Uncertainty for every planet at once, using the archive's own error columns
unc = eh.compute_esi_mc_catalog(df, method="2p", n_samples=2000, seed=0)

# Plots (needs exohab[viz] or exohab[all])
eh.plot_esi_scatter(df[df.pl_rade < 2.5])
eh.plot_habitable_zone(df, color_col="esi_2p")
eh.plot_mc_distribution(mc)

# scikit-learn feature engineering (needs exohab[ml] or exohab[all])
X = eh.HabitabilityFeatureTransformer().fit_transform(df)
```

## What is included

| Area | Functions | Definition used |
|---|---|---|
| Earth Similarity Index | `compute_esi_4p` | Schulze-Makuch et al. (2011): `ESI_x = (1 - abs(x-x0)/(x+x0))**w`; weights radius 0.57, density 1.07, escape velocity 0.70, surface temperature 5.58; interior, surface and global indices as geometric means |
| Two-parameter ESI | `compute_esi_2p` | Planetary Habitability Laboratory: `1 - sqrt(0.5*[((S-1)/(S+1))**2 + ((R-1)/(R+1))**2])` |
| Similarity to other bodies | `compute_similarity` | Same form with other reference values. `"mars"` is the published Mars Similarity Index (Kashyap Jagadeesh et al. 2017). `"venus"` and `"titan"` use NASA fact-sheet values with the Earth ESI weights, because no body-specific weights are published. Custom dicts are supported. |
| Planetary Habitability Index | `compute_phi` | `(S*E*C*L)**(1/4)` (Schulze-Makuch et al. 2011); you supply factor scores in [0, 1] |
| Biological Complexity Index | `compute_bci` | `(S*E*T*G*A)**(1/5)`, optional `BCI_rel = BCI/max` (Irwin et al. 2014); you supply factor scores in [0, 1] |
| Surface temperature | `estimate_surface_temp` | `T_s = 9.650 + 1.096*T_eq` (Kashyap Jagadeesh 2017), only for rocky planets with atmospheres |
| Habitable zone | `kopparapu_seff`, `hz_distance` | Kopparapu et al. (2014) flux limits (recent Venus, runaway greenhouse for 0.1/1/5 Earth masses, maximum greenhouse, early Mars), valid for 2600–7200 K; `d = sqrt(L/S_eff)` |
| Uncertainty | `compute_esi_mc`, `compute_esi_2p_mc`, `compute_esi_mc_catalog`, `sample_split_normal`, `split_normal_cdf` | Two-piece (split) normal for `x +a -b` quotes; optional truncation at zero; density and escape velocity derived per draw from mass and radius to keep them correlated |
| Data | `load_nasa_exoplanets`, `add_habitability_columns` | NASA Exoplanet Archive TAP, `pscomppars` table |
| ML | `HabitabilityFeatureTransformer` | 15 features: derived density and escape velocity, ESI scores, interior/surface ratio, HZ fluxes and distances, in-HZ flags |
| Plots | `plot_esi_scatter`, `plot_habitable_zone`, `plot_mc_distribution` | matplotlib |

All metric functions accept scalars, lists, NumPy arrays and pandas Series.
Series indices are preserved, and missing or invalid values return NaN instead
of raising, so a whole catalogue can be processed in one call.

## Scientific notes and limitations

Please read these before using results in research:

- **Surface vs. equilibrium temperature.** The 4-parameter ESI is defined with
  mean *surface* temperature (Earth 288 K), but catalogues list *equilibrium*
  temperature (Earth about 255 K). By default `pl_eqt` is used directly, which
  lowers the surface ESI of temperate planets. `surface_temp_method="kashyap2017"`
  applies an empirical fit to five Solar System bodies. It is not valid for gas
  giants, and it extrapolates above about 255 K.
- **Mass.** `pl_bmasse` is the archive's "best mass", which may be a minimum
  mass (M sin i) for radial-velocity planets.
- **ESI is a similarity measure, not a probability of life.** A high ESI means
  Earth-like bulk properties and insolation, nothing more.
- **PHI and BCI need expert factor scores.** `exohab` implements only the
  published combination formulas; it does not invent scores from raw data.
- **Monte Carlo inputs are sampled independently**, apart from the
  mass–radius correlation that is kept when `mass` is given. Missing error
  bars are treated as zero and flagged (`*_errors_complete`), so those
  intervals are lower limits.
- **Kopparapu HZ limits** are NaN outside 2600–7200 K unless
  `extrapolate=True` is passed explicitly.

## Data source and acknowledgement

Live data comes from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
No API key is needed. If you publish results that use it, the archive asks you
to include this acknowledgement:

> This research has made use of the NASA Exoplanet Archive, which is operated
> by the California Institute of Technology, under contract with the National
> Aeronautics and Space Administration under the Exoplanet Exploration Program.

**Behind a corporate proxy or antivirus HTTPS scanning?** If you see
`SSL: CERTIFICATE_VERIFY_FAILED`, run `pip install truststore`, then call
`import truststore; truststore.inject_into_ssl()` before loading data. This uses
your operating system's certificate store. Do not disable certificate
verification.

## References

- Schulze-Makuch, D. et al. (2011). A Two-Tiered Approach to Assessing the Habitability of Exoplanets. *Astrobiology* 11(10), 1041–1052. doi:10.1089/ast.2010.0592
- Planetary Habitability Laboratory, UPR Arecibo. Earth Similarity Index. https://phl.upr.edu/projects/earth-similarity-index-esi
- Kashyap Jagadeesh, M. (2017). *Earth Similarity Index and Habitability Studies of Exoplanets.* M.Phil. dissertation, Christ University. arXiv:1801.07101
- Irwin, L. N., Méndez, A., Fairén, A. G., Schulze-Makuch, D. (2014). Assessing the Possibility of Biological Complexity on Other Worlds. *Challenges* 5(1), 159–174. doi:10.3390/challe5010159
- Kopparapu, R. K. et al. (2013). Habitable Zones around Main-sequence Stars: New Estimates. *ApJ* 765, 131. arXiv:1301.6674
- Kopparapu, R. K. et al. (2014). Habitable Zones around Main-sequence Stars: Dependence on Planetary Mass. *ApJL* 787, L29. arXiv:1404.5292
- Wallis, K. F. (2014). The Two-Piece Normal, Binormal, or Double Gaussian Distribution. *Statistical Science* 29(1), 106–112.

## Citing exohab

If `exohab` helps your work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff), and cite the original papers of the metrics you
use (listed above).

## License

MIT — see [LICENSE](LICENSE).
