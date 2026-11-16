---
description: Compute Kopparapu et al. (2014) habitable zone boundaries - effective flux and orbital distance - for any star in Python with exohab.
---

# Habitable zone

exohab implements the habitable-zone boundaries of Kopparapu et al. (2014).
Each boundary is given as an effective stellar flux \(S_\mathrm{eff}\)
(Earth = 1) that depends on the star's effective temperature.

| Limit | Meaning | Zone |
|---|---|---|
| `recent_venus` | Venus lost its water | Optimistic inner edge |
| `runaway_greenhouse` | Oceans start to evaporate | **Conservative inner edge** |
| `maximum_greenhouse` | CO₂ can no longer warm the planet | **Conservative outer edge** |
| `early_mars` | Early Mars may have had liquid water | Optimistic outer edge |
| `runaway_greenhouse_5me`, `runaway_greenhouse_0.1me` | Runaway greenhouse for 5 and 0.1 Earth-mass planets | Inner edge variants |

```python
import exohab as eh

inner = eh.kopparapu_seff(5780.0, "runaway_greenhouse")   # 1.107
outer = eh.kopparapu_seff(5780.0, "maximum_greenhouse")   # 0.356

# Convert flux to distance (AU): d = sqrt(L / S_eff), L in solar luminosities
print(eh.hz_distance(1.0, inner), eh.hz_distance(1.0, outer))   # ~0.95 AU, ~1.68 AU
```

The NASA archive gives luminosity as \(\log_{10}(L/L_\odot)\) (column
`st_lum`). Pass `log_luminosity=True` in that case:

```python
import exohab as eh

d = eh.hz_distance(-0.932, eh.kopparapu_seff(4402.0), log_luminosity=True)   # Kepler-442
```

!!! warning "Valid range"
    The fits are valid for stars between 2600 K and 7200 K. Outside that range
    exohab returns NaN instead of extrapolating silently. Pass
    `extrapolate=True` to `kopparapu_seff` only if you accept that risk.

Plot planets against the habitable zone with
[`plot_habitable_zone`](plots.md#habitable-zone-diagram).
