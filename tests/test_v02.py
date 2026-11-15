"""Tests for features added in v0.2.0."""

import io
import math
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import exohab
from exohab import (
    compute_esi_2p,
    compute_esi_2p_mc,
    compute_esi_4p,
    compute_esi_mc,
    compute_esi_mc_catalog,
    estimate_surface_temp,
    sample_split_normal,
    split_normal_cdf,
)


@pytest.fixture
def raw(sample_csv):
    return pd.read_csv(io.StringIO(sample_csv))


def test_version_matches_installed_metadata():
    assert exohab.__version__ == "0.2.0" == version("exohab")


# --------------------------------------------------------------------------- surface temp
class TestSurfaceTemp:
    def test_kashyap_relation(self):
        # Kashyap Jagadeesh (2017) eq. 3.12: T_s = 9.650 + 1.096 T_eq
        assert estimate_surface_temp(255.0) == pytest.approx(9.650 + 1.096 * 255.0)
        # source Table 3.3 (computed with unrounded coefficients): Earth 289.28, Mars 247.61,
        # Titan 99.57, Triton 47.15 -> agreement within coefficient rounding
        for teq, published in [(255, 289.28), (217, 247.61), (82, 99.57), (34.2, 47.15)]:
            assert estimate_surface_temp(teq) == pytest.approx(published, abs=0.2)

    def test_recovers_earth_far_better_than_raw_teq(self):
        # Using T_eq directly underestimates Earth's surface ESI (~0.84);
        # the estimated surface temperature (289.1 K) nearly recovers 1.0
        raw_s = compute_esi_4p(1, 1, 1, 255.0).surface
        est_s = compute_esi_4p(1, 1, 1, estimate_surface_temp(255.0)).surface
        assert raw_s < 0.85 < 0.99 < est_s

    def test_invalid_and_types(self):
        assert math.isnan(estimate_surface_temp(-10))
        assert math.isnan(estimate_surface_temp(0))
        s = estimate_surface_temp(pd.Series([255.0, np.nan], index=["a", "b"]))
        assert list(s.index) == ["a", "b"] and np.isnan(s["b"])
        with pytest.raises(KeyError):
            estimate_surface_temp(255.0, method="made_up")

    def test_loader_option(self, raw):
        plain = exohab.add_habitability_columns(raw)
        est = exohab.add_habitability_columns(raw, surface_temp_method="kashyap2017")
        assert "surface_temp_est" not in plain.columns
        assert est["surface_temp_est"].iloc[2] == pytest.approx(9.650 + 1.096 * 241.0)
        assert "kashyap2017" in est.attrs["esi_4p_temperature"]
        # interior ESI does not depend on temperature
        pd.testing.assert_series_equal(plain["esi_4p_interior"], est["esi_4p_interior"])
        assert not np.allclose(plain["esi_4p_surface"].dropna(), est["esi_4p_surface"].dropna())

    def test_transformer_option(self, raw):
        pytest.importorskip("sklearn")
        t = exohab.HabitabilityFeatureTransformer(surface_temp_method="kashyap2017")
        out = t.fit_transform(raw)
        ref = exohab.add_habitability_columns(raw, surface_temp_method="kashyap2017")
        np.testing.assert_allclose(out["esi_4p"], ref["esi_4p"])


# --------------------------------------------------------------------------- truncation
class TestTruncation:
    def test_cdf_properties(self):
        assert split_normal_cdf(1.0, 1.0, 0.3, 0.1) == pytest.approx(0.1 / 0.4)
        x = np.linspace(-2, 4, 200)
        f = split_normal_cdf(x, 1.0, 0.3, 0.1)
        assert np.all(np.diff(f) >= 0) and f[0] == pytest.approx(0, abs=1e-12)
        assert f[-1] == pytest.approx(1, abs=1e-12)
        # symmetric case equals the normal CDF
        assert split_normal_cdf(1.0 + 1.0, 1.0, 1.0, 1.0) == pytest.approx(0.841344746, abs=1e-8)
        # empirical CDF of the sampler matches the analytic CDF
        rng = np.random.default_rng(0)
        s = sample_split_normal(1.0, 0.3, 0.1, 200_000, rng)
        for q in (0.85, 0.95, 1.0, 1.2, 1.5):
            assert np.mean(s <= q) == pytest.approx(split_normal_cdf(q, 1.0, 0.3, 0.1), abs=0.005)

    def test_truncated_sampler(self):
        rng = np.random.default_rng(1)
        s = sample_split_normal(0.1, 0.1, 0.1, 50_000, rng, lower=0.0)
        assert s.size == 50_000 and np.all(s > 0)
        # truncated normal mean: mu + sigma*phi(a)/(1-Phi(a)), a = (0-mu)/sigma = -1
        phi = math.exp(-0.5) / math.sqrt(2 * math.pi)
        expected = 0.1 + 0.1 * phi / (1 - 0.158655254)
        assert s.mean() == pytest.approx(expected, abs=0.002)

    def test_truncation_is_noop_when_bound_irrelevant(self):
        a = sample_split_normal(10.0, 1.0, 1.0, 1000, np.random.default_rng(3))
        b = sample_split_normal(10.0, 1.0, 1.0, 1000, np.random.default_rng(3), lower=0.0)
        assert np.array_equal(a, b)

    def test_inconsistent_bound_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            sample_split_normal(-5.0, 0.1, 0.1, 10, lower=0.0)
        with pytest.raises(ValueError):
            sample_split_normal(0.0, 0.0, 0.0, 10, lower=0.0)

    def test_mc_truncate_keeps_sample_size(self):
        plain = compute_esi_2p_mc((0.1, 0.1), 1.0, n_samples=5000, seed=4)
        trunc = compute_esi_2p_mc((0.1, 0.1), 1.0, n_samples=5000, seed=4, truncate=True)
        assert plain.n_valid < 5000 and trunc.n_valid == 5000
        assert np.all(trunc.samples > 0)
        res = compute_esi_mc((1, 0.5), surface_temp=(288, 10), mass=(1, 0.9),
                             n_samples=3000, seed=2, truncate=True)
        assert res.n_valid == 3000


# --------------------------------------------------------------------------- catalogue MC
class TestCatalogMC:
    def test_2p_matches_single_planet_function(self, raw):
        out = compute_esi_mc_catalog(raw, "2p", n_samples=20_000, seed=0)
        assert out.index.equals(raw.index)
        k = 2  # Kepler-442 b row: R 1.34 +0.11/-0.18, S 0.66 (no S errors in fixture)
        single = compute_esi_2p_mc((1.34, 0.11, 0.18), 0.66, n_samples=20_000, seed=0)
        assert out.loc[k, "esi_2p_mc_median"] == pytest.approx(single.median, abs=0.003)
        assert out.loc[k, "esi_2p_mc_ci68_low"] == pytest.approx(single.ci68[0], abs=0.005)
        assert not out.loc[k, "esi_2p_mc_errors_complete"]  # insolation errors missing

    def test_zero_errors_give_point_value(self):
        df = pd.DataFrame({"pl_rade": [1.12, 2.0], "pl_radeerr1": [0, 0], "pl_radeerr2": [0, 0],
                           "pl_insol": [0.700522, 1.0], "pl_insolerr1": [0, 0],
                           "pl_insolerr2": [0, 0]})
        out = compute_esi_mc_catalog(df, "2p", n_samples=50, seed=1)
        np.testing.assert_allclose(out["esi_2p_mc_median"], compute_esi_2p(df.pl_rade, df.pl_insol))
        np.testing.assert_allclose(out["esi_2p_mc_std"], 0, atol=1e-12)
        assert out["esi_2p_mc_errors_complete"].all()
        assert (out["esi_2p_mc_n_valid"] == 50).all()

    def test_4p_point_value_and_surface_temp_option(self):
        df = pd.DataFrame({"pl_rade": [1.2], "pl_bmasse": [1.8], "pl_eqt": [250.0]})
        out = compute_esi_mc_catalog(df, "4p", n_samples=20, seed=0)
        exp = compute_esi_4p(1.2, 1.8 / 1.2**3, math.sqrt(1.8 / 1.2), 250.0).esi
        assert out.loc[0, "esi_4p_mc_median"] == pytest.approx(exp)
        out2 = compute_esi_mc_catalog(df, "4p", n_samples=20, seed=0,
                                      surface_temp_method="kashyap2017")
        exp2 = compute_esi_4p(1.2, 1.8 / 1.2**3, math.sqrt(1.8 / 1.2),
                              estimate_surface_temp(250.0)).esi
        assert out2.loc[0, "esi_4p_mc_median"] == pytest.approx(exp2)

    def test_missing_rows_and_chunking(self, raw):
        a = compute_esi_mc_catalog(raw, "2p", n_samples=500, seed=9, chunk_size=1)
        empty = raw.index[raw.pl_name == "Synthetic empty"][0]
        assert a.loc[empty, "esi_2p_mc_n_valid"] == 0
        assert a.loc[empty, ["esi_2p_mc_median", "esi_2p_mc_ci95_low"]].isna().all()
        valid = a["esi_2p_mc_n_valid"] > 0
        assert (a.loc[valid, "esi_2p_mc_ci95_low"] <= a.loc[valid, "esi_2p_mc_median"]).all()
        assert (a.loc[valid, "esi_2p_mc_median"] <= a.loc[valid, "esi_2p_mc_ci95_high"]).all()
        b = compute_esi_mc_catalog(raw, "2p", n_samples=500, seed=9, chunk_size=1)
        pd.testing.assert_frame_equal(a, b)  # reproducible

    def test_errors(self, raw):
        with pytest.raises(ValueError):
            compute_esi_mc_catalog(raw, "3p")
        with pytest.raises(KeyError):
            compute_esi_mc_catalog(raw.drop(columns=["pl_insol"]), "2p")


# --------------------------------------------------------------------------- viz
class TestVizV02:
    def test_habitable_zone_plot(self, raw):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = exohab.add_habitability_columns(raw)
        ax = exohab.plot_habitable_zone(df, labels=["Kepler-442 b"], color_col="esi_2p")
        assert ax.get_xscale() == "log"
        lo, hi = ax.get_xlim()
        assert lo > hi  # inverted insolation axis
        assert any(t.get_text() == "Kepler-442 b" for t in ax.texts)
        assert len(ax.collections) >= 4  # two HZ bands, planets, Earth
        with pytest.raises(KeyError):
            exohab.plot_habitable_zone(df, flux_col="nope")
        plt.close("all")

    def test_scatter_labels_do_not_overlap(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # many candidates crowded near Earth
        rng = np.random.default_rng(0)
        n = 12
        df = pd.DataFrame({"esi_4p_interior": 0.97 + rng.uniform(0, 0.02, n),
                           "esi_4p_surface": 0.96 + rng.uniform(0, 0.03, n),
                           "pl_name": [f"P{i}" for i in range(n)]})
        ax = exohab.plot_esi_scatter(df, esi_col=None, top_n=n)
        ys = sorted(t.xyann[1] for t in ax.texts if t.get_text().startswith("P"))
        assert len(ys) == n
        assert np.min(np.diff(ys)) >= 0.045 - 1e-9
        assert ys[0] >= 0 and ys[-1] <= 1.05
        plt.close("all")
