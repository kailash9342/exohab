"""Offline tests for the NASA Exoplanet Archive loader (HTTP is mocked)."""

import io
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from exohab import compute_esi_2p, compute_esi_4p, data, load_nasa_exoplanets
from exohab.data import TAP_URL, add_habitability_columns, build_query, default_cache_dir


class FakeResponse:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class Recorder:
    """Stand-in for requests.get that records calls."""

    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        if self.exc is not None:
            raise self.exc
        return self.response


@pytest.fixture
def fake_get(monkeypatch, sample_csv):
    rec = Recorder(FakeResponse(sample_csv))
    monkeypatch.setattr(data.requests, "get", rec)
    return rec


def test_default_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("EXOHAB_CACHE_DIR", str(tmp_path / "x"))
    assert default_cache_dir() == tmp_path / "x"
    monkeypatch.delenv("EXOHAB_CACHE_DIR")
    assert default_cache_dir() == Path.home() / ".exohab" / "cache"


def test_build_query():
    q = build_query(["pl_name"], where="pl_rade < 2")
    assert q.startswith("select pl_name,") and " from pscomppars where pl_rade < 2" in q
    for required in ("pl_rade", "pl_bmasse", "pl_insol", "pl_eqt"):
        assert required in q
    with pytest.raises(ValueError):
        build_query(["pl_name; drop table"])


def test_fetch_parses_and_derives(fake_get):
    df = load_nasa_exoplanets()
    url, params, _ = fake_get.calls[0]
    assert url == TAP_URL
    assert params["format"] == "csv" and "pscomppars" in params["query"]
    assert len(df) == 5

    row = df.set_index("pl_name").loc["Kepler-442 b"]
    assert row["density_eu"] == pytest.approx(2.36 / 1.34**3)
    assert row["escape_velocity_eu"] == pytest.approx(math.sqrt(2.36 / 1.34))
    # mass-radius density agrees with the archive's own pl_dens (g/cm^3)
    assert row["density_eu"] * 5.51 == pytest.approx(row["pl_dens"], rel=0.01)
    assert row["esi_2p"] == pytest.approx(compute_esi_2p(1.34, 0.66))
    expected = compute_esi_4p(1.34, row["density_eu"], row["escape_velocity_eu"], 241.0)
    assert row["esi_4p"] == pytest.approx(expected.esi)
    assert row["esi_4p_interior"] == pytest.approx(expected.interior)


def test_fallback_without_mass(fake_get):
    df = load_nasa_exoplanets().set_index("pl_name")
    syn = df.loc["Synthetic no-mass"]
    # density from pl_dens / 5.51 = 0.5; v_e = R * sqrt(rho) = 2 * sqrt(0.5)
    assert syn["density_eu"] == pytest.approx(0.5)
    assert syn["escape_velocity_eu"] == pytest.approx(2 * math.sqrt(0.5))
    # consistency: implied mass rho*R^3 = 4 gives sqrt(M/R) = sqrt(2) as well
    assert syn["escape_velocity_eu"] == pytest.approx(math.sqrt(4 / 2))
    empty = df.loc["Synthetic empty"]
    for c in ("density_eu", "escape_velocity_eu", "esi_2p", "esi_4p"):
        assert np.isnan(empty[c])


def test_cache_is_used_offline(fake_get, monkeypatch):
    first = load_nasa_exoplanets()
    cache_files = list(default_cache_dir().glob("pscomppars_*.csv"))
    assert len(cache_files) == 1

    offline = Recorder(exc=requests.ConnectionError("offline"))
    monkeypatch.setattr(data.requests, "get", offline)
    second = load_nasa_exoplanets()
    assert offline.calls == []
    pd.testing.assert_frame_equal(first, second)


def test_different_queries_use_different_cache_files(fake_get):
    load_nasa_exoplanets()
    load_nasa_exoplanets(where="pl_rade < 2")
    assert len(fake_get.calls) == 2
    assert len(list(default_cache_dir().glob("pscomppars_*.csv"))) == 2


def test_refresh_and_max_age(fake_get):
    load_nasa_exoplanets()
    load_nasa_exoplanets(refresh=True)
    assert len(fake_get.calls) == 2
    path = next(default_cache_dir().glob("pscomppars_*.csv"))
    old = time.time() - 10 * 86400
    os.utime(path, (old, old))
    load_nasa_exoplanets(max_age_days=30)
    assert len(fake_get.calls) == 2  # still fresh enough
    load_nasa_exoplanets(max_age_days=1)
    assert len(fake_get.calls) == 3  # stale -> re-downloaded


def test_network_failure_falls_back_to_cache(fake_get, monkeypatch):
    load_nasa_exoplanets()
    monkeypatch.setattr(data.requests, "get", Recorder(exc=requests.ConnectionError("down")))
    with pytest.warns(RuntimeWarning, match="cached data"):
        df = load_nasa_exoplanets(refresh=True)
    assert len(df) == 5


def test_network_failure_without_cache_raises(monkeypatch):
    monkeypatch.setattr(data.requests, "get", Recorder(exc=requests.ConnectionError("down")))
    with pytest.raises(requests.ConnectionError):
        load_nasa_exoplanets()


def test_http_error_and_votable_error(monkeypatch):
    monkeypatch.setattr(data.requests, "get", Recorder(FakeResponse("", status=500)))
    with pytest.raises(requests.HTTPError):
        load_nasa_exoplanets()
    err = '<?xml version="1.0"?><VOTABLE><INFO name="QUERY_STATUS" value="ERROR"/></VOTABLE>'
    monkeypatch.setattr(data.requests, "get", Recorder(FakeResponse(err)))
    with pytest.raises(RuntimeError, match="error document"):
        load_nasa_exoplanets()
    assert not list(default_cache_dir().glob("*.csv"))  # errors are never cached


def test_session_and_raw_mode(sample_csv):
    class Session:
        def __init__(self):
            self.get = Recorder(FakeResponse(sample_csv))

    s = Session()
    df = load_nasa_exoplanets(session=s, add_metrics=False)
    assert len(s.get.calls) == 1
    assert "esi_2p" not in df.columns


def test_add_habitability_columns_does_not_mutate(sample_csv):
    raw = pd.read_csv(io.StringIO(sample_csv))
    before = raw.copy()
    out = add_habitability_columns(raw)
    pd.testing.assert_frame_equal(raw, before)
    assert {"density_eu", "escape_velocity_eu", "esi_2p", "esi_4p"} <= set(out.columns)
    valid = out["esi_2p"].dropna()
    assert ((valid >= 0) & (valid <= 1)).all()
