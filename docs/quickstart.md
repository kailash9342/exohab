---
description: Quick start for exohab - compute the Earth Similarity Index, propagate uncertainties and build machine-learning features for exoplanets in Python.
---

# Quick start

Pick the path that fits you. Every snippet runs as-is after
`pip install "exohab[all]"`.

## :material-school: For students: compare worlds to Earth

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

## :material-microscope: For researchers: propagate asymmetric uncertainties

```python
import exohab as eh

# Kepler-442 b: R = 1.34 +0.11/-0.18 R_E, S = 0.66 +0.23/-0.41 S_E (NASA Exoplanet Archive)
mc = eh.compute_esi_2p_mc(radius=(1.34, 0.11, -0.18), flux=(0.66, 0.23, -0.41),
                          n_samples=20_000, seed=42, truncate=True)
print(mc.median, mc.ci68, mc.ci95)

# TRAPPIST-1 e, four-parameter ESI. Density and escape velocity are derived for
# each draw from mass and radius, so their correlation is kept.
ts = eh.estimate_surface_temp(249.7)                     # ~283.3 K
mc4 = eh.compute_esi_mc(radius=(0.920, 0.013, -0.012), mass=(0.692, 0.022, -0.022),
                        surface_temp=(ts, 1.096 * 2.4), n_samples=20_000, seed=42)
print(mc4.median, mc4.components["interior"].median, mc4.components["surface"].median)

# Uncertainty for the whole catalogue, from the archive's own err1/err2 columns
df = eh.load_nasa_exoplanets(surface_temp_method="kashyap2017")
unc = eh.compute_esi_mc_catalog(df, method="2p", n_samples=2000, seed=0)
print(df.join(unc).nlargest(10, "esi_2p_mc_median")[["pl_name", "esi_2p", "esi_2p_mc_median"]])
```

## :material-robot: For ML engineers: features in a scikit-learn pipeline

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

## Next steps

- Read the [user guide](guide/metrics.md) for each feature in depth.
- Check [Science & methods](science.md) for the exact formulas and their limitations.
- Browse the [API reference](api.md) for every function and parameter.
