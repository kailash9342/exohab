---
description: Propagate asymmetric exoplanet measurement errors into the Earth Similarity Index with Monte Carlo sampling (split normal) in Python using exohab.
---

# Uncertainty (Monte Carlo)

Exoplanet radii, masses and fluxes come with error bars that are often
lopsided, such as \(1.34^{+0.11}_{-0.18}\). A single ESI value hides that.
exohab samples every input many times and gives you the resulting ESI
**distribution**.

## How to give a measurement

| Form | Meaning |
|---|---|
| `1.34` | Exact value, no uncertainty |
| `(1.34, 0.1)` | Symmetric 1-sigma error |
| `(1.34, 0.11, -0.18)` | Asymmetric errors, upper and lower. The sign of the lower error is ignored, so archive `err1`/`err2` pairs can be used directly |

Asymmetric errors are sampled from the two-piece ("split") normal
distribution, the standard reading of a \(x^{+a}_{-b}\) quote.

## One planet

```python
import exohab as eh

mc = eh.compute_esi_2p_mc(radius=(1.34, 0.11, -0.18), flux=(0.66, 0.23, -0.41),
                          n_samples=20_000, seed=42, truncate=True)
print(mc.median, mc.ci68, mc.ci95, mc.n_valid)
```

The result is an `MCResult` with `mean`, `median`, `std`, `ci68`, `ci95`, the
raw `samples` and `n_valid`. Call `mc.interval(0.9)` for other levels.

For the four-parameter ESI, pass `mass` instead of density and escape
velocity. Both are then derived from the same mass and radius in each draw,
which keeps their physical correlation:

```python
import exohab as eh

mc4 = eh.compute_esi_mc(radius=(0.920, 0.013, -0.012), mass=(0.692, 0.022, -0.022),
                        surface_temp=(283.3, 2.6), n_samples=20_000, seed=42)
print(mc4.median, mc4.components["interior"].median, mc4.components["surface"].median)
```

## Negative draws: `truncate`

Large error bars can produce physically impossible draws, such as a negative
flux.

- `truncate=False` (default): those draws are discarded, and `n_valid`
  reports how many remain.
- `truncate=True`: those draws are redrawn, so all `n_samples` are kept.

Both describe the same distribution truncated at zero.

## The whole catalogue

```python
import exohab as eh

df = eh.load_nasa_exoplanets()
unc = eh.compute_esi_mc_catalog(df, method="2p", n_samples=2000, seed=0)
print(unc.columns.tolist())
```

The result has one row per planet with `esi_2p_mc_median`, the 68% and 95%
interval bounds, `esi_2p_mc_n_valid` and `esi_2p_mc_errors_complete`.

!!! info "Missing error bars"
    When the archive gives no error for a value, it is treated as exact, and
    `*_errors_complete` is `False`. For those planets the interval is a
    **lower limit** on the true uncertainty.

## Sampling tools

`sample_split_normal` and `split_normal_cdf` are public, if you need the
distribution directly:

```python
import numpy as np
import exohab as eh

draws = eh.sample_split_normal(1.0, 0.3, 0.1, 10_000, np.random.default_rng(0))
print(eh.split_normal_cdf(1.0, 1.0, 0.3, 0.1))   # 0.25 of the mass lies below the mode
```
