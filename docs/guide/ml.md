---
description: Engineer exoplanet habitability features for machine learning with a scikit-learn compatible transformer from exohab.
---

# Machine-learning features

`HabitabilityFeatureTransformer` is a scikit-learn transformer. It turns raw
NASA Exoplanet Archive columns into 15 habitability features. It needs
`pip install "exohab[ml]"` or `"exohab[all]"`.

```python
import exohab as eh

df = eh.load_nasa_exoplanets()
t = eh.HabitabilityFeatureTransformer(surface_temp_method="kashyap2017")
X = t.fit_transform(df)
print(X.shape)        # (number of planets, 15)
```

## Features

| Feature | Meaning |
|---|---|
| `density_eu`, `escape_velocity_eu` | Derived from mass and radius |
| `esi_2p`, `esi_4p_interior`, `esi_4p_surface`, `esi_4p` | ESI scores |
| `esi_interior_surface_ratio` | Interior ESI divided by surface ESI |
| `hz_seff_inner`, `hz_seff_outer` | Conservative habitable-zone flux limits |
| `hz_seff_inner_optimistic`, `hz_seff_outer_optimistic` | Optimistic limits |
| `hz_inner_au`, `hz_outer_au` | Conservative limits as distances |
| `in_conservative_hz`, `in_optimistic_hz` | 1.0 inside, 0.0 outside, NaN if unknown |

Habitable-zone features are NaN for stars outside 2600–7200 K, where the
Kopparapu fits are not valid.

## In a pipeline

```python
import exohab as eh
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

df = eh.load_nasa_exoplanets()
model = make_pipeline(
    eh.HabitabilityFeatureTransformer(),
    SimpleImputer(strategy="median"),
    KMeans(n_clusters=4, n_init="auto", random_state=0),
)
df["cluster"] = model.fit_predict(df)
```

## Options

- Column names default to the NASA archive names (`pl_rade`, `pl_bmasse`,
  `pl_insol`, `pl_eqt`, `st_teff`, `st_lum`) and can all be changed.
- `keep_original=True` returns your input columns followed by the features.
- The input must be a pandas DataFrame.
- The transformer is stateless: `fit` only checks the input.
