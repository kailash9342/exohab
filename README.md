# exohab

Fetch live exoplanet data, compute planetary habitability metrics, propagate
measurement uncertainties and plot results — without the boilerplate.

```bash
pip install exohab          # core: numpy, pandas, scipy, requests
pip install "exohab[all]"   # + matplotlib and scikit-learn
```

```python
import exohab as eh

eh.compute_esi_2p(radius=1.12, flux=0.70)            # Proxima b -> ~0.87
eh.compute_esi_4p(0.53, 0.71, 0.45, 240.0)           # Mars -> interior/surface/global

mc = eh.compute_esi_2p_mc(radius=(1.34, 0.11, 0.18), flux=(0.66, 0.23, 0.23), seed=1)
print(mc.median, mc.ci68)

df = eh.load_nasa_exoplanets()                       # cached in ~/.exohab/cache
```

Full documentation is coming in a later release. Every formula used is cited in
the docstring of the function that implements it.
