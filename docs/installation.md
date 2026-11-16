---
description: How to install exohab, the Python exoplanet habitability library, with pip, including optional plotting and machine-learning extras.
---

# Installation

exohab needs **Python 3.10 or newer**.

=== "Everything (recommended)"

    ```bash
    pip install "exohab[all]"
    ```

    Includes plotting (matplotlib) and machine-learning features (scikit-learn).

=== "Core only"

    ```bash
    pip install exohab
    ```

    Depends only on `numpy`, `pandas` and `requests`.

=== "Pick extras"

    ```bash
    pip install "exohab[viz]"   # + matplotlib
    pip install "exohab[ml]"    # + scikit-learn
    ```

## Check the installation

```python
import exohab as eh

print(eh.__version__)
print(eh.compute_esi_4p(radius=1.0, density=1.0, escape_velocity=1.0, surface_temp=288.0))
# ESIResult(interior=1.0, surface=1.0, esi=1.0)
```

For a complete check of every function, run the
[full tour script](https://github.com/kailash9342/exohab/blob/main/examples/full_tour.py).
It compares results with published values and should end with `26 passed`.

## Optional extras are imported lazily

`import exohab` never fails because matplotlib or scikit-learn are missing.
You only get a clear `ImportError` (with the install command) when you call a
plotting function or create the ML transformer.

## Behind a proxy or antivirus HTTPS scanning?

If loading NASA data fails with `SSL: CERTIFICATE_VERIFY_FAILED`, let Python
use your operating system's certificate store:

```bash
pip install truststore
```

```python
import truststore
truststore.inject_into_ssl()
```

Run that before loading data. Please do **not** disable certificate
verification.

## Development install

```bash
git clone https://github.com/kailash9342/exohab.git
cd exohab
pip install -e ".[dev,all]"
pytest
```

All tests run offline; network calls are mocked.
