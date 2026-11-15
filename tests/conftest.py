"""Shared fixtures. All tests run offline."""

import os

import pytest

try:  # headless backend for plotting tests (matplotlib is optional)
    import matplotlib

    matplotlib.use("Agg")
except ImportError:  # pragma: no cover
    pass

# Real rows (subset of columns) as returned by the NASA Exoplanet Archive
# pscomppars table, queried 2026-11-15, plus one synthetic row with missing
# mass to exercise the density/escape-velocity fallbacks.
SAMPLE_CSV = """pl_name,hostname,pl_rade,pl_radeerr1,pl_radeerr2,pl_bmasse,pl_dens,pl_insol,pl_eqt,st_teff,st_lum
"TRAPPIST-1 e","TRAPPIST-1",0.92000000,0.01300000,-0.01200000,0.69200000,4.901605,0.6460,249.70,2566.00,-3.25727
"TRAPPIST-1 b","TRAPPIST-1",1.11600000,0.01400000,-0.01200000,1.37400000,5.441939,4.1530,397.60,2566.00,-3.25727
"Kepler-442 b","Kepler-442",1.34000000,0.11000000,-0.18000000,2.36000000,5.390000,0.6600,241.00,4402.00,-0.93200
"Synthetic no-mass","Synth",2.00000000,,,,2.755000,1.0000,288.00,5780.00,0.0
"Synthetic empty","Synth",,,,,,,,,
"""


@pytest.fixture
def sample_csv():
    return SAMPLE_CSV


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Never touch the real ~/.exohab cache during tests."""
    monkeypatch.setenv("EXOHAB_CACHE_DIR", str(tmp_path / "cache"))
    yield
    os.environ.pop("EXOHAB_CACHE_DIR", None)
