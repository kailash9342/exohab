"""Tests for Kopparapu et al. (2014) habitable-zone boundaries."""

import math

import numpy as np
import pandas as pd
import pytest

from exohab import KOPPARAPU_2014, hz_distance, kopparapu_seff


def test_solar_values_equal_seffsun():
    # At Teff = 5780 K the polynomial reduces to S_eff_sun (hzcalc.py)
    expected = {
        "recent_venus": 1.776,
        "runaway_greenhouse": 1.107,
        "maximum_greenhouse": 0.356,
        "early_mars": 0.320,
        "runaway_greenhouse_5me": 1.188,
        "runaway_greenhouse_0.1me": 0.99,
    }
    for limit, val in expected.items():
        assert kopparapu_seff(5780.0, limit) == pytest.approx(val, abs=1e-12)


def test_polynomial_matches_manual_evaluation():
    s0, a, b, c, d = KOPPARAPU_2014["maximum_greenhouse"]
    t = 3300.0 - 5780.0
    manual = s0 + a * t + b * t**2 + c * t**3 + d * t**4
    assert kopparapu_seff(3300.0, "maximum_greenhouse") == pytest.approx(manual, rel=1e-14)


def test_ordering_of_limits_over_valid_range():
    teff = np.linspace(2600, 7200, 47)
    rv = kopparapu_seff(teff, "recent_venus")
    rg = kopparapu_seff(teff, "runaway_greenhouse")
    mg = kopparapu_seff(teff, "maximum_greenhouse")
    em = kopparapu_seff(teff, "early_mars")
    assert np.all(rv > rg) and np.all(rg > mg) and np.all(mg > em) and np.all(em > 0)
    # cooler stars -> lower boundary fluxes (more red/IR light, lower albedo)
    assert np.all(np.diff(rg) > 0) and np.all(np.diff(mg) > 0)


def test_heavier_planet_has_closer_runaway_limit():
    teff = np.linspace(2600, 7200, 10)
    assert np.all(
        kopparapu_seff(teff, "runaway_greenhouse_5me") > kopparapu_seff(teff, "runaway_greenhouse")
    )
    assert np.all(
        kopparapu_seff(teff, "runaway_greenhouse_0.1me") < kopparapu_seff(teff, "runaway_greenhouse")
    )


def test_validity_range():
    out = kopparapu_seff([2500.0, 2600.0, 7200.0, 7300.0, np.nan])
    assert np.isnan(out[0]) and np.isfinite(out[1]) and np.isfinite(out[2])
    assert np.isnan(out[3]) and np.isnan(out[4])
    assert np.isfinite(kopparapu_seff(7300.0, extrapolate=True))
    with pytest.raises(KeyError):
        kopparapu_seff(5000.0, "moist_greenhouse_typo")


def test_distance_sun():
    # d = sqrt(L / S_eff): Sun -> inner 1/sqrt(1.107) AU, outer 1/sqrt(0.356) AU
    assert hz_distance(1.0, 1.107) == pytest.approx(1 / math.sqrt(1.107))
    assert hz_distance(1.0, 0.356) == pytest.approx(1 / math.sqrt(0.356))
    # inverse-square law: 4x luminosity -> 2x distance
    assert hz_distance(4.0, 1.0) == pytest.approx(2.0)
    # log luminosity input as in NASA's st_lum
    assert hz_distance(np.log10(4.0), 1.0, log_luminosity=True) == pytest.approx(2.0)


def test_distance_invalid_and_series():
    assert math.isnan(hz_distance(0.0, 1.0))
    assert math.isnan(hz_distance(1.0, -1.0))
    s = hz_distance(pd.Series([1.0, 4.0], index=["a", "b"]), 1.0)
    assert list(s.index) == ["a", "b"] and s["b"] == pytest.approx(2.0)
