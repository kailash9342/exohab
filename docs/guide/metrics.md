---
description: Compute the Earth Similarity Index (ESI), Mars Similarity Index, PHI and BCI for exoplanets in Python with exohab, including custom reference bodies.
---

# Habitability metrics

All metric functions accept scalars, lists, NumPy arrays and pandas Series.
Series keep their index. Missing or invalid values (negative, NaN, `None`)
give NaN instead of raising an error, so a whole catalogue can be processed in
one call.

## Earth Similarity Index: four parameters

`compute_esi_4p` uses radius, bulk density and escape velocity (all in Earth
units) and the mean **surface** temperature in kelvin. It returns an
`ESIResult` with the interior, surface and global index.

```python
import exohab as eh

earth = eh.compute_esi_4p(radius=1.0, density=1.0, escape_velocity=1.0, surface_temp=288.0)
print(earth)        # ESIResult(interior=1.0, surface=1.0, esi=1.0)

mars = eh.compute_esi_4p(radius=0.53, density=0.71, escape_velocity=0.45, surface_temp=240.0)
print(round(mars.esi, 2))   # 0.73 (Kashyap Jagadeesh 2017)
```

Only have mass and radius? Derive the other two quantities:

```python
import exohab as eh

rho = eh.density_from_mass_radius(mass=0.107, radius=0.532)            # ~0.71 Earth units
ve = eh.escape_velocity_from_mass_radius(mass=0.107, radius=0.532)     # ~0.45 Earth units
```

Catalogues list equilibrium temperature, not surface temperature. You can
estimate one from the other with `estimate_surface_temp` (rocky planets with
an atmosphere only; see [Science & methods](../science.md#surface-temperature)).

```python
import exohab as eh

ts = eh.estimate_surface_temp(255.0)    # Earth's T_eq -> ~289 K
```

## Earth Similarity Index: two parameters

For transiting planets that have only a radius and a stellar flux, the
Planetary Habitability Laboratory uses a two-parameter form:

```python
import exohab as eh

eh.compute_esi_2p(radius=1.12, flux=0.700522)        # Proxima b -> 0.8692
eh.compute_esi_2p(radius=[1.0, 0.532, 11.2], flux=[1.0, 0.431, 0.037])   # Earth, Mars, Jupiter
```

## Similarity to other worlds

`compute_similarity` uses the same ESI formula with another body's reference
values. The built-in presets are in `REFERENCE_BODIES`:

| Preset | Reference values | Weights |
|---|---|---|
| `"earth"` | 1 EU, 288 K | Schulze-Makuch et al. (2011) |
| `"mars"` | 3390 km, 3.93 g cm⁻³, 5.03 km s⁻¹, 240 K | Mars Similarity Index weights (Kashyap Jagadeesh et al. 2017) |
| `"venus"` | NASA Planetary Fact Sheet | Earth ESI weights (no body-specific weights published) |
| `"titan"` | NASA Saturnian Satellite Fact Sheet | Earth ESI weights (no body-specific weights published) |

```python
import exohab as eh

msi = eh.compute_similarity(1.0, 1.0, 1.0, 288.0, reference="mars")
print(round(msi.esi, 2))    # Earth on the Mars scale -> 0.68

# A custom reference body (Earth units, surface temperature in K)
ref = {"radius": 1.5, "density": 1.0, "escape_velocity": 1.2, "surface_temp": 300.0}
res = eh.compute_similarity(1.4, 1.0, 1.1, 290.0, reference=ref)
```

Inputs are always in **Earth units**, also for non-Earth references.

## Planetary Habitability Index and Biological Complexity Index

PHI and BCI combine factor scores between 0 and 1 that you assign using the
rubrics in the original papers. exohab implements only the published way of
combining them; it does not invent scores from raw data.

```python
import exohab as eh

# Illustrative scores only, not published values
phi = eh.compute_phi(substrate=0.8, energy=0.9, chemistry=0.7, liquid=0.6)
bci = eh.compute_bci(substrate=[0.9, 0.5], energy=[0.8, 0.5], temperature=[0.7, 0.4],
                     geophysics=[0.9, 0.3], age=[1.0, 0.6], relative=True)
```

With `relative=True`, BCI values are divided by the largest one in the set, as
in Irwin et al. (2014).
