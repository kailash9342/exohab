"""Tests for the optional ML and visualisation helpers and their import guards."""

import builtins
import io
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import exohab
from exohab import compute_esi_2p_mc, kopparapu_seff


@pytest.fixture
def raw(sample_csv):
    return pd.read_csv(io.StringIO(sample_csv))


# --------------------------------------------------------------------------- ML
class TestTransformer:
    def test_features_and_values(self, raw):
        pytest.importorskip("sklearn")
        from exohab.ml import FEATURE_NAMES

        t = exohab.HabitabilityFeatureTransformer()
        out = t.fit_transform(raw)
        assert list(out.columns) == FEATURE_NAMES
        assert out.index.equals(raw.index)
        k = out.loc[raw["pl_name"] == "Kepler-442 b"].iloc[0]
        assert k["hz_seff_inner"] == pytest.approx(kopparapu_seff(4402.0, "runaway_greenhouse"))
        assert k["hz_seff_outer"] == pytest.approx(kopparapu_seff(4402.0, "maximum_greenhouse"))
        assert k["esi_interior_surface_ratio"] == pytest.approx(
            k["esi_4p_interior"] / k["esi_4p_surface"]
        )
        lum = 10**-0.932
        assert k["hz_inner_au"] == pytest.approx(np.sqrt(lum / k["hz_seff_inner"]))
        # S = 0.66 lies between the conservative limits for a 4402 K star
        assert k["hz_seff_outer"] < 0.66 < k["hz_seff_inner"]
        assert k["in_conservative_hz"] == 1.0

    def test_hz_flags(self, raw):
        pytest.importorskip("sklearn")
        t = exohab.HabitabilityFeatureTransformer()
        out = t.fit_transform(raw)
        names = raw["pl_name"]
        # TRAPPIST-1 (Teff = 2566 K) is outside the 2600-7200 K validity of the
        # Kopparapu fits -> HZ features are NaN rather than extrapolated
        trap = out.loc[names.str.startswith("TRAPPIST")]
        assert trap[["hz_seff_inner", "hz_inner_au", "in_conservative_hz"]].isna().all().all()
        assert trap["esi_2p"].notna().all()
        empty = out.loc[names == "Synthetic empty"].iloc[0]
        assert np.isnan(empty["in_conservative_hz"]) and np.isnan(empty["esi_4p"])

        # Sun-like star: S = 1 inside both HZs; S = 4.15 inside neither; S = 0.33
        # (between early-Mars 0.320 and max-greenhouse 0.356) only optimistic
        sun = pd.DataFrame(
            {"pl_rade": [1.0, 1.0, 1.0], "pl_insol": [1.0, 4.15, 0.33],
             "st_teff": [5780.0] * 3, "st_lum": [0.0] * 3}
        )
        f = t.fit_transform(sun)
        assert list(f["in_conservative_hz"]) == [1.0, 0.0, 0.0]
        assert list(f["in_optimistic_hz"]) == [1.0, 0.0, 1.0]
        assert f.loc[0, "hz_inner_au"] == pytest.approx(1 / np.sqrt(1.107))

    def test_sklearn_api(self, raw):
        pytest.importorskip("sklearn")
        from sklearn.base import clone
        from sklearn.exceptions import NotFittedError
        from sklearn.pipeline import Pipeline

        t = exohab.HabitabilityFeatureTransformer(keep_original=True)
        assert clone(t).get_params()["keep_original"] is True
        with pytest.raises(NotFittedError):
            exohab.HabitabilityFeatureTransformer().transform(raw)
        out = Pipeline([("hab", t)]).fit_transform(raw)
        assert list(out.columns[: raw.shape[1]]) == list(raw.columns)
        assert list(t.get_feature_names_out()) == list(out.columns)
        with pytest.raises(TypeError):
            exohab.HabitabilityFeatureTransformer().fit(raw.to_numpy())
        with pytest.raises(KeyError):
            exohab.HabitabilityFeatureTransformer(radius_col="nope").fit(raw)

    def test_missing_sklearn_message(self, monkeypatch):
        import exohab.ml as ml

        monkeypatch.setattr(ml, "_HAS_SKLEARN", False)
        with pytest.raises(ImportError, match=r"pip install exohab\[all\]"):
            ml.HabitabilityFeatureTransformer()


# --------------------------------------------------------------------------- viz
class TestViz:
    def test_scatter(self, raw):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = exohab.add_habitability_columns(raw)
        ax = exohab.plot_esi_scatter(df, top_n=2)
        texts = {t.get_text() for t in ax.texts}
        top2 = set(df.nlargest(2, "esi_4p")["pl_name"])
        assert top2 <= texts
        # one scatter point per row with finite sub-indices (+ Earth marker)
        n_valid = df[["esi_4p_interior", "esi_4p_surface"]].dropna().shape[0]
        assert ax.collections[0].get_offsets().shape[0] == n_valid
        plt.close("all")

    def test_scatter_without_global_column(self, raw):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = exohab.add_habitability_columns(raw).drop(columns=["esi_4p"])
        exohab.plot_esi_scatter(df, esi_col="esi_4p", top_n=0)
        with pytest.raises(KeyError):
            exohab.plot_esi_scatter(df, interior_col="missing")
        plt.close("all")

    def test_mc_distribution(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        res = compute_esi_2p_mc((1.3, 0.1), (0.8, 0.1), n_samples=2000, seed=0)
        ax = exohab.plot_mc_distribution(res)
        median_lines = [l for l in ax.lines if np.allclose(l.get_xdata(), res.median)]
        assert median_lines
        assert len(ax.patches) >= 60  # histogram bars
        with pytest.raises(TypeError):
            exohab.plot_mc_distribution([0.1, 0.2])
        plt.close("all")

    def test_missing_matplotlib_message(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        res = compute_esi_2p_mc(1.0, 1.0, n_samples=10)
        with pytest.raises(ImportError, match=r"pip install exohab\[all\]"):
            exohab.plot_mc_distribution(res)


# --------------------------------------------------------------------------- imports
def test_version_and_exports():
    assert isinstance(exohab.__version__, str)
    for name in exohab.__all__:
        assert hasattr(exohab, name), name
    with pytest.raises(AttributeError):
        exohab.does_not_exist  # noqa: B018


def test_import_without_optional_dependencies():
    # Fresh interpreter where matplotlib and sklearn imports are blocked.
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('matplotlib', 'sklearn'):\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import exohab\n"
        "assert exohab.compute_esi_2p(1, 1) == 1.0\n"
        "assert 'matplotlib' not in sys.modules and 'sklearn' not in sys.modules\n"
        "try:\n"
        "    exohab.HabitabilityFeatureTransformer()\n"
        "except ImportError as e:\n"
        "    assert 'exohab[all]' in str(e)\n"
        "else:\n"
        "    raise SystemExit('no ImportError')\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
