"""Tests for Monte Carlo error propagation."""

import math

import numpy as np
import pytest

from exohab import (
    MCResult,
    compute_esi_2p,
    compute_esi_2p_mc,
    compute_esi_4p,
    compute_esi_mc,
    sample_split_normal,
)
from exohab.uncertainty import parse_measurement


class TestParse:
    def test_forms(self):
        assert parse_measurement(1.5) == (1.5, 0.0, 0.0)
        assert parse_measurement((1.5, 0.2)) == (1.5, 0.2, 0.2)
        # NASA err2 values are negative; sign is ignored
        assert parse_measurement((1.34, 0.11, -0.18)) == (1.34, 0.11, 0.18)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_measurement((1, 2, 3, 4))
        with pytest.raises(ValueError):
            parse_measurement((np.nan, 0.1))


class TestSplitNormal:
    def test_zero_error_is_constant(self):
        s = sample_split_normal(3.0, 0, 0, 100)
        assert np.all(s == 3.0)

    def test_symmetric_matches_normal_moments(self):
        rng = np.random.default_rng(0)
        s = sample_split_normal(10.0, 2.0, 2.0, 200_000, rng)
        assert s.mean() == pytest.approx(10.0, abs=0.02)
        assert s.std() == pytest.approx(2.0, rel=0.01)

    def test_asymmetric_theoretical_properties(self):
        # Two-piece normal: P(x < mode) = s_lo/(s_lo+s_hi), mean = mode + sqrt(2/pi)(s_hi - s_lo)
        rng = np.random.default_rng(1)
        mode, hi, lo = 1.0, 0.3, 0.1
        s = sample_split_normal(mode, hi, lo, 400_000, rng)
        assert np.mean(s < mode) == pytest.approx(lo / (lo + hi), abs=0.005)
        assert s.mean() == pytest.approx(mode + math.sqrt(2 / math.pi) * (hi - lo), abs=0.002)
        # each half is a half-normal with its own sigma
        assert np.sqrt(np.mean((s[s > mode] - mode) ** 2)) == pytest.approx(hi, rel=0.01)
        assert np.sqrt(np.mean((s[s < mode] - mode) ** 2)) == pytest.approx(lo, rel=0.01)


class TestESIMC:
    def test_exact_inputs_reproduce_deterministic_value(self):
        res = compute_esi_mc(0.53, 0.71, 0.45, 240.0, n_samples=50, seed=0)
        det = compute_esi_4p(0.53, 0.71, 0.45, 240.0)
        # zero spread (up to floating-point summation error)
        assert res.std == pytest.approx(0.0, abs=1e-12) and res.n_valid == 50
        assert res.mean == pytest.approx(det.esi) and res.median == pytest.approx(det.esi)
        assert res.ci68 == pytest.approx((det.esi, det.esi))
        assert res.components["interior"].median == pytest.approx(det.interior)
        assert res.components["surface"].median == pytest.approx(det.surface)

    def test_mass_path_equals_direct_path_without_errors(self):
        a = compute_esi_mc(1.2, surface_temp=270.0, mass=1.8, n_samples=10, seed=0)
        rho = 1.8 / 1.2**3
        ve = math.sqrt(1.8 / 1.2)
        assert a.median == pytest.approx(compute_esi_4p(1.2, rho, ve, 270.0).esi)

    def test_summary_statistics_consistent(self):
        res = compute_esi_mc(
            (1.34, 0.11, -0.18), (0.97, 0.2), (1.32, 0.15), (265.0, 15.0, 10.0),
            n_samples=20_000, seed=42,
        )
        s = res.samples
        assert res.n_valid == s.size and s.size <= 20_000
        assert np.all((s > 0) & (s <= 1))
        assert res.ci95[0] <= res.ci68[0] <= res.median <= res.ci68[1] <= res.ci95[1]
        assert res.mean == pytest.approx(s.mean())
        assert res.std == pytest.approx(s.std(ddof=1))
        assert res.ci95 == pytest.approx(tuple(np.percentile(s, [2.5, 97.5])))
        assert res.interval(0.5)[0] < res.median < res.interval(0.5)[1]

    def test_earth_with_errors_is_biased_below_one(self):
        # ESI has its maximum at Earth, so any scatter can only lower it
        res = compute_esi_mc((1, 0.05), (1, 0.05), (1, 0.05), (288, 5), n_samples=5000, seed=3)
        assert res.samples.max() <= 1.0
        assert res.mean < 1.0 and res.median < 1.0
        assert res.mean > 0.9

    def test_wider_errors_give_wider_intervals(self):
        narrow = compute_esi_2p_mc((1.5, 0.05), (0.8, 0.05), n_samples=20_000, seed=5)
        wide = compute_esi_2p_mc((1.5, 0.3), (0.8, 0.3), n_samples=20_000, seed=5)
        assert (wide.ci68[1] - wide.ci68[0]) > (narrow.ci68[1] - narrow.ci68[0])
        assert wide.std > narrow.std

    def test_reproducible_with_seed(self):
        a = compute_esi_2p_mc((1.1, 0.1, 0.2), (0.7, 0.1), n_samples=1000, seed=7)
        b = compute_esi_2p_mc((1.1, 0.1, 0.2), (0.7, 0.1), n_samples=1000, seed=7)
        c = compute_esi_2p_mc((1.1, 0.1, 0.2), (0.7, 0.1), n_samples=1000, seed=8)
        assert np.array_equal(a.samples, b.samples)
        assert not np.array_equal(a.samples, c.samples)

    def test_2p_converges_to_deterministic_for_small_errors(self):
        res = compute_esi_2p_mc((1.12, 1e-4), (0.700522, 1e-4), n_samples=2000, seed=0)
        assert res.median == pytest.approx(compute_esi_2p(1.12, 0.700522), abs=1e-4)

    def test_nonphysical_draws_are_discarded(self):
        # 1-sigma error comparable to the value -> some negative radius draws
        res = compute_esi_2p_mc((0.1, 0.1), 1.0, n_samples=10_000, seed=11)
        assert 0 < res.n_valid < 10_000
        assert res.n_samples == 10_000
        assert np.all(np.isfinite(res.samples))

    def test_argument_validation(self):
        with pytest.raises(ValueError):
            compute_esi_mc(1.0, surface_temp=288.0)  # no density / mass
        with pytest.raises(ValueError):
            compute_esi_mc(1.0, 1.0, None, 288.0)
        with pytest.raises(ValueError):
            compute_esi_mc(1.0, 1.0, 1.0, 288.0, mass=1.0)
        with pytest.raises(ValueError):
            compute_esi_mc(1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            compute_esi_2p_mc(1.0, 1.0, n_samples=1)

    def test_empty_result(self):
        r = MCResult.from_samples(np.array([np.nan, np.nan]))
        assert r.n_valid == 0 and math.isnan(r.mean) and math.isnan(r.interval(0.9)[0])
        d = compute_esi_2p_mc(1.0, 1.0, n_samples=10).to_dict()
        assert d["median"] == 1.0 and d["n_valid"] == 10
