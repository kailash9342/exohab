"""Tests for exohab.metrics against published values and mathematical properties."""

import math

import numpy as np
import pandas as pd
import pytest

from exohab import (
    ESI_WEIGHTS,
    REFERENCE_BODIES,
    compute_bci,
    compute_esi_2p,
    compute_esi_4p,
    compute_phi,
    compute_similarity,
    density_from_mass_radius,
    escape_velocity_from_mass_radius,
)

EARTH = (1.0, 1.0, 1.0, 288.0)


def _term(x, x0, w):
    """Independent re-implementation of the single-property ESI term."""
    return (1 - abs(x - x0) / (x + x0)) ** w


# --------------------------------------------------------------------------- ESI 4p
class TestESI4P:
    def test_published_weights(self):
        # Schulze-Makuch et al. (2011) Table 1 / Kashyap (2017) Table 3.2
        assert ESI_WEIGHTS == {
            "radius": 0.57,
            "density": 1.07,
            "escape_velocity": 0.70,
            "surface_temp": 5.58,
        }

    def test_earth_is_exactly_one(self):
        res = compute_esi_4p(*EARTH)
        assert res.interior == 1.0 and res.surface == 1.0 and res.esi == 1.0

    def test_mars_worked_example_kashyap_2017(self):
        # arXiv:1801.07101 p.23: R=0.53, rho=0.71, ve=0.45, Ts=240 K
        # -> ESI_I = 0.82, ESI_S = 0.65, ESI = 0.73
        res = compute_esi_4p(0.53, 0.71, 0.45, 240.0)
        assert res.interior == pytest.approx(0.82, abs=0.005)
        assert res.surface == pytest.approx(0.65, abs=0.005)
        assert res.esi == pytest.approx(0.73, abs=0.005)

    @pytest.mark.parametrize(
        "name, params, expected",
        [
            # Kashyap (2017) Table 3.4 (inputs and outputs rounded to 2 d.p.)
            ("Kepler-442 b", (1.34, 0.97, 1.32, 265.0), (0.94, 0.84, 0.89)),
            ("GJ 667C c", (1.54, 1.05, 1.57, 280.0), (0.92, 0.88, 0.90)),
            ("Kepler-62 e", (1.61, 1.08, 1.68, 296.0), (0.90, 0.87, 0.88)),
        ],
    )
    def test_published_exoplanet_table(self, name, params, expected):
        res = compute_esi_4p(*params)
        # tolerance reflects 2-decimal rounding of the published inputs/outputs
        for got, want in zip(res, expected):
            assert got == pytest.approx(want, abs=0.015), name

    def test_matches_independent_formula(self):
        r, d, v, t = 0.8, 1.3, 0.9, 250.0
        si = math.sqrt(_term(r, 1, 0.57) * _term(d, 1, 1.07))
        ss = math.sqrt(_term(v, 1, 0.70) * _term(t, 288, 5.58))
        res = compute_esi_4p(r, d, v, t)
        assert res.interior == pytest.approx(si, rel=1e-12)
        assert res.surface == pytest.approx(ss, rel=1e-12)
        assert res.esi == pytest.approx(math.sqrt(si * ss), rel=1e-12)

    def test_equivalent_to_phl_product_form(self):
        # PHL: ESI = prod (1-|..|)^(w_i/n), n = 4
        x = (1.7, 0.6, 1.2, 320.0)
        refs = (1, 1, 1, 288)
        ws = (0.57, 1.07, 0.70, 5.58)
        phl = np.prod([_term(xi, x0, w / 4) for xi, x0, w in zip(x, refs, ws)])
        assert compute_esi_4p(*x).esi == pytest.approx(phl, rel=1e-12)

    def test_ratio_symmetry(self):
        # |x - x0|/(x + x0) is identical for x = k*x0 and x = x0/k
        a = compute_esi_4p(2.0, 1, 1, 288).esi
        b = compute_esi_4p(0.5, 1, 1, 288).esi
        assert a == pytest.approx(b, rel=1e-12)

    def test_monotonic_away_from_earth(self):
        radii = np.array([1.0, 1.2, 1.5, 2.0, 4.0, 11.0])
        esi = compute_esi_4p(radii, 1.0, 1.0, 288.0).esi
        assert np.all(np.diff(esi) < 0)

    def test_temperature_dominates_weighting(self):
        # Same fractional deviation: temperature (w=5.58) penalises more than radius (w=0.57)
        hot = compute_esi_4p(1, 1, 1, 288 * 1.2).esi
        big = compute_esi_4p(1.2, 1, 1, 288).esi
        assert hot < big

    def test_bounds_on_random_inputs(self):
        rng = np.random.default_rng(0)
        r = rng.uniform(0.01, 30, 5000)
        d = rng.uniform(0.01, 5, 5000)
        v = rng.uniform(0.01, 10, 5000)
        t = rng.uniform(1, 3000, 5000)
        res = compute_esi_4p(r, d, v, t)
        for arr in res:
            assert arr.shape == (5000,)
            assert np.all((arr > 0) & (arr <= 1))
        assert np.allclose(res.esi, np.sqrt(res.interior * res.surface))

    def test_zero_gives_zero_and_negative_gives_nan(self):
        assert compute_esi_4p(0.0, 1, 1, 288).esi == 0.0
        assert math.isnan(compute_esi_4p(-1.0, 1, 1, 288).esi)
        assert math.isnan(compute_esi_4p(1.0, 1, 1, -5).surface)

    def test_nan_propagates_only_to_affected_subindex(self):
        res = compute_esi_4p(1.0, 1.0, np.nan, 288.0)
        assert res.interior == 1.0
        assert math.isnan(res.surface) and math.isnan(res.esi)

    def test_input_types(self):
        lst = compute_esi_4p([1.0, 0.53], [1.0, 0.71], [1.0, 0.45], [288, 240])
        assert isinstance(lst.esi, np.ndarray) and lst.esi.shape == (2,)
        assert isinstance(compute_esi_4p(*EARTH).esi, float)
        idx = pd.Index(["Earth", "Mars"])
        s = compute_esi_4p(
            pd.Series([1.0, 0.53], index=idx), [1.0, 0.71], [1.0, 0.45], [288, 240]
        )
        assert isinstance(s.esi, pd.Series)
        assert list(s.esi.index) == ["Earth", "Mars"]
        assert s.esi["Mars"] == pytest.approx(lst.esi[1])

    def test_2d_broadcasting(self):
        r = np.linspace(0.5, 2, 4)[:, None]
        t = np.linspace(200, 350, 3)[None, :]
        assert compute_esi_4p(r, 1.0, 1.0, t).esi.shape == (4, 3)

    def test_custom_weights_change_result(self):
        base = compute_esi_4p(1.5, 1, 1, 288).esi
        heavier = compute_esi_4p(1.5, 1, 1, 288, weights={"radius": 2.0}).esi
        assert heavier < base
        with pytest.raises(ValueError):
            compute_esi_4p(1, 1, 1, 288, weights={"radius": 0.0})


# --------------------------------------------------------------------------- ESI 2p
class TestESI2P:
    def test_earth(self):
        assert compute_esi_2p(1.0, 1.0) == 1.0

    def test_proxima_b_published(self):
        # arXiv:1712.01040: R=1.12, S=0.700522 -> ESI = 0.8692
        assert compute_esi_2p(1.12, 0.700522) == pytest.approx(0.8692, abs=5e-5)

    def test_mars_about_064(self):
        # Mars: R = 3389.5/6371 = 0.532 R_E, S = 1/1.5237^2 = 0.431 S_E
        r = 3389.5 / 6371.0
        s = 1 / 1.52371**2
        assert compute_esi_2p(r, s) == pytest.approx(0.645, abs=0.005)

    def test_formula(self):
        r, s = 2.3, 0.4
        dr, ds = (r - 1) / (r + 1), (s - 1) / (s + 1)
        assert compute_esi_2p(r, s) == pytest.approx(1 - math.sqrt(0.5 * (dr**2 + ds**2)))

    def test_symmetric_in_parameters(self):
        assert compute_esi_2p(1.7, 0.6) == pytest.approx(compute_esi_2p(0.6, 1.7))

    def test_edge_cases(self):
        assert compute_esi_2p(0.0, 0.0) == pytest.approx(0.0, abs=1e-15)
        assert math.isnan(compute_esi_2p(-1, 1))
        assert math.isnan(compute_esi_2p(1, np.nan))
        assert math.isnan(compute_esi_2p(None, 1))
        out = compute_esi_2p(np.array([1.0, -2.0, np.inf]), 1.0)
        assert out[0] == 1.0 and np.isnan(out[1:]).all()

    def test_bounds_and_series(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.uniform(0, 25, 1000))
        s = pd.Series(rng.uniform(0, 5000, 1000))
        e = compute_esi_2p(r, s)
        assert isinstance(e, pd.Series) and len(e) == 1000
        assert ((e >= 0) & (e <= 1)).all()


# --------------------------------------------------------------------------- similarity
class TestSimilarity:
    def test_earth_preset_equals_esi(self):
        x = (1.3, 0.9, 1.1, 270)
        assert compute_similarity(*x, reference="earth") == compute_esi_4p(*x)

    def test_reference_body_is_one_against_itself(self):
        for name, body in REFERENCE_BODIES.items():
            res = compute_similarity(
                body["radius"], body["density"], body["escape_velocity"], body["surface_temp"],
                reference=name,
            )
            assert res.esi == pytest.approx(1.0, abs=1e-12), name

    def test_earth_msi_published(self):
        # Kashyap (2017) Table 3.5: Earth MSI_I = 0.70, MSI_S = 0.66, MSI = 0.68
        res = compute_similarity(*EARTH, reference="Mars")
        assert res.interior == pytest.approx(0.70, abs=0.01)
        assert res.surface == pytest.approx(0.66, abs=0.005)
        assert res.esi == pytest.approx(0.68, abs=0.005)

    def test_custom_reference_dict(self):
        ref = {"radius": 2.0, "density": 1.0, "escape_velocity": 1.4, "surface_temp": 300.0}
        res = compute_similarity(2.0, 1.0, 1.4, 300.0, reference=ref)
        assert res.esi == pytest.approx(1.0)
        # default weights are the ESI weights
        r = compute_similarity(1.0, 1.0, 1.4, 300.0, reference=ref).interior
        assert r == pytest.approx(math.sqrt(_term(1.0, 2.0, 0.57)))

    def test_errors(self):
        with pytest.raises(KeyError):
            compute_similarity(*EARTH, reference="pluto")
        with pytest.raises(KeyError):
            compute_similarity(*EARTH, reference={"radius": 1.0})
        with pytest.raises(ValueError):
            compute_similarity(*EARTH, reference={**REFERENCE_BODIES["earth"], "radius": -1})

    def test_venus_vs_earth_driven_by_temperature(self):
        v = REFERENCE_BODIES["venus"]
        res = compute_esi_4p(v["radius"], v["density"], v["escape_velocity"], v["surface_temp"])
        assert res.interior > 0.95  # Venus is Earth's twin in bulk properties
        assert res.surface < 0.3  # ...but not at the surface (737 K)


# --------------------------------------------------------------------------- derived
def test_derived_quantities():
    assert density_from_mass_radius(1, 1) == 1.0
    assert density_from_mass_radius(8, 2) == 1.0
    assert escape_velocity_from_mass_radius(4, 1) == 2.0
    assert math.isnan(density_from_mass_radius(1, 0))
    assert math.isnan(escape_velocity_from_mass_radius(-1, 1))
    # Mars: M = 0.107 M_E, R = 0.532 R_E -> rho ~ 0.71, ve ~ 0.45 (Kashyap 2017)
    assert density_from_mass_radius(0.107, 0.532) == pytest.approx(0.71, abs=0.01)
    assert escape_velocity_from_mass_radius(0.107, 0.532) == pytest.approx(0.45, abs=0.01)


# --------------------------------------------------------------------------- PHI / BCI
class TestPHIandBCI:
    def test_phi_geometric_mean(self):
        assert compute_phi(1, 1, 1, 1) == 1.0
        assert compute_phi(0.5, 0.5, 0.5, 0.5) == pytest.approx(0.5)
        assert compute_phi(0.2, 0.4, 0.6, 0.8) == pytest.approx((0.2 * 0.4 * 0.6 * 0.8) ** 0.25)
        assert compute_phi(0, 1, 1, 1) == 0.0

    def test_phi_bounded_by_min_and_max(self):
        rng = np.random.default_rng(2)
        f = rng.uniform(0, 1, (4, 500))
        phi = compute_phi(*f)
        assert np.all(phi >= f.min(axis=0) - 1e-12) and np.all(phi <= f.max(axis=0) + 1e-12)

    def test_phi_invalid(self):
        assert math.isnan(compute_phi(1.2, 1, 1, 1))
        assert math.isnan(compute_phi(-0.1, 1, 1, 1))
        assert math.isnan(compute_phi(np.nan, 1, 1, 1))

    def test_bci_abs_and_rel(self):
        assert compute_bci(1, 1, 1, 1, 1) == 1.0
        assert compute_bci(0.5, 0.5, 0.5, 0.5, 0.5) == pytest.approx(0.5)
        vals = compute_bci([0.9, 0.4], [0.8, 0.5], [0.7, 0.6], [0.9, 0.2], [1.0, 0.3])
        rel = compute_bci([0.9, 0.4], [0.8, 0.5], [0.7, 0.6], [0.9, 0.2], [1.0, 0.3], relative=True)
        assert rel[0] == pytest.approx(1.0)
        assert rel[1] == pytest.approx(vals[1] / vals[0])

    def test_bci_relative_ignores_invalid_rows(self):
        rel = compute_bci([0.5, 2.0], 0.5, 0.5, 0.5, 0.5, relative=True)
        assert rel[0] == pytest.approx(1.0) and np.isnan(rel[1])
        assert np.isnan(compute_bci([0.0], 1, 1, 1, 1, relative=True)).all()
