---
description: Load every confirmed exoplanet from the NASA Exoplanet Archive into pandas with exohab - no API key, offline cache, and habitability columns added automatically.
---

# NASA Exoplanet Archive data

`load_nasa_exoplanets` downloads the Planetary Systems Composite Parameters
table (`pscomppars`, one row per confirmed planet) from the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) TAP
service. No API key is needed.

```python
import exohab as eh

df = eh.load_nasa_exoplanets()
print(len(df))
print(df.nlargest(5, "esi_2p")[["pl_name", "pl_rade", "pl_insol", "esi_2p", "esi_4p"]])
```

## Caching and offline use

The first call downloads the data; later calls read a local copy in
`~/.exohab/cache` and work offline.

| Option | Effect |
|---|---|
| `refresh=True` | Download again now |
| `max_age_days=7` | Download again if the cache is older than 7 days |
| `cache_dir="..."` | Use another folder (or set the `EXOHAB_CACHE_DIR` environment variable) |

If a refresh fails because you're offline, the cached copy is used with a
warning.

## Filtering on the server

```python
import exohab as eh

small_near = eh.load_nasa_exoplanets(where="pl_rade < 2 and sy_dist < 50")
```

`where` is an ADQL condition, run by the archive itself.

## Columns added

| Column | Meaning |
|---|---|
| `density_eu` | Bulk density in Earth units: \(M/R^3\), or `pl_dens`/5.51 when mass is missing |
| `escape_velocity_eu` | \(\sqrt{M/R}\), or \(R\sqrt{\rho}\) when mass is missing |
| `esi_2p` | Two-parameter ESI from radius and insolation |
| `esi_4p_interior`, `esi_4p_surface`, `esi_4p` | Four-parameter ESI |
| `surface_temp_est` | Only with `surface_temp_method="kashyap2017"` |

!!! warning "Equilibrium vs. surface temperature"
    By default the archive's equilibrium temperature `pl_eqt` is used directly
    as the temperature input of the four-parameter ESI. That lowers the score
    of temperate planets (Earth's equilibrium temperature is about 255 K, its
    surface 288 K). Use `surface_temp_method="kashyap2017"` to convert it first.
    The temperature used is recorded in `df.attrs["esi_4p_temperature"]`.

```python
import exohab as eh

df = eh.load_nasa_exoplanets(surface_temp_method="kashyap2017")
print(df.attrs["esi_4p_temperature"])
```

You can also add the same columns to your own table with
`add_habitability_columns(df, ...)`.

## Acknowledgement

If you publish results based on this data, the archive asks you to include:

> This research has made use of the NASA Exoplanet Archive, which is operated
> by the California Institute of Technology, under contract with the National
> Aeronautics and Space Administration under the Exoplanet Exploration Program.
