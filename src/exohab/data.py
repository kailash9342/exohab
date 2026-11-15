"""Live data from the NASA Exoplanet Archive (no API key required).

Queries the Table Access Protocol (TAP) synchronous endpoint for the
Planetary Systems Composite Parameters table (``pscomppars``: one row per
confirmed planet, best available value per column) and caches the raw CSV in
``~/.exohab/cache`` (override with ``EXOHAB_CACHE_DIR`` or ``cache_dir``).

Column meanings follow the archive's documentation:
``pl_rade`` radius [R_earth], ``pl_bmasse`` best mass [M_earth] (true mass or
M sin i), ``pl_dens`` density [g/cm^3], ``pl_insol`` insolation [S_earth],
``pl_eqt`` equilibrium temperature [K], ``st_teff`` [K], ``st_lum``
[log10(L/L_sun)].
"""

from __future__ import annotations

import hashlib
import io
import os
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests

from .metrics import (
    EARTH_DENSITY_GCC,
    compute_esi_2p,
    compute_esi_4p,
    density_from_mass_radius,
    escape_velocity_from_mass_radius,
)

__all__ = [
    "TAP_URL",
    "DEFAULT_COLUMNS",
    "default_cache_dir",
    "build_query",
    "add_habitability_columns",
    "load_nasa_exoplanets",
]

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

DEFAULT_COLUMNS: tuple = (
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "pl_orbper",
    "pl_orbsmax",
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_bmasse",
    "pl_bmasseerr1",
    "pl_bmasseerr2",
    "pl_dens",
    "pl_insol",
    "pl_insolerr1",
    "pl_insolerr2",
    "pl_eqt",
    "pl_eqterr1",
    "pl_eqterr2",
    "st_teff",
    "st_lum",
    "st_mass",
    "st_rad",
    "sy_dist",
)

_REQUIRED = ("pl_name", "pl_rade", "pl_bmasse", "pl_dens", "pl_insol", "pl_eqt")


def default_cache_dir() -> Path:
    """Cache directory: ``$EXOHAB_CACHE_DIR`` or ``~/.exohab/cache``."""
    env = os.environ.get("EXOHAB_CACHE_DIR")
    return Path(env).expanduser() if env else Path.home() / ".exohab" / "cache"


def build_query(columns: Optional[Iterable[str]] = None, where: Optional[str] = None) -> str:
    """Build the ADQL query sent to the TAP service.

    Parameters
    ----------
    columns : iterable of str, optional
        Columns to select; the columns needed for the ESI are always added.
    where : str, optional
        ADQL ``WHERE`` clause body, e.g. ``"pl_rade < 2 and sy_dist < 50"``.

    Returns
    -------
    str
        ADQL query string.
    """
    cols = list(DEFAULT_COLUMNS if columns is None else columns)
    for c in _REQUIRED:
        if c not in cols:
            cols.append(c)
    for c in cols:
        if not c.replace("_", "").isalnum():
            raise ValueError(f"invalid column name: {c!r}")
    q = f"select {','.join(cols)} from pscomppars"
    if where:
        q += f" where {where}"
    return q


def add_habitability_columns(
    df: pd.DataFrame,
    radius_col: str = "pl_rade",
    mass_col: str = "pl_bmasse",
    density_col: Optional[str] = "pl_dens",
    flux_col: str = "pl_insol",
    surface_temp_col: str = "pl_eqt",
) -> pd.DataFrame:
    """Append derived Earth-unit properties and ESI columns.

    Added columns:

    * ``density_eu`` - ``M / R**3`` when mass and radius exist, otherwise
      ``pl_dens / 5.51``.
    * ``escape_velocity_eu`` - ``sqrt(M / R)``; if mass is missing but radius
      and density exist, ``R * sqrt(rho_eu)`` (same physics: ``M = rho R^3``).
    * ``esi_2p`` - radius + insolation ESI (:func:`exohab.compute_esi_2p`).
    * ``esi_4p_interior``, ``esi_4p_surface``, ``esi_4p`` - 4-parameter ESI.

    .. warning::
       Exoplanet catalogues list *equilibrium* temperature, not surface
       temperature. By default ``pl_eqt`` is used as the temperature input of
       the 4-parameter ESI, without any greenhouse correction, against the
       288 K reference. This is a proxy and systematically lowers the surface
       ESI of temperate planets (Earth: T_eq ~ 255 K). Supply a column of
       modelled surface temperatures via ``surface_temp_col`` if available.

    Parameters
    ----------
    df : DataFrame
        Input table (not modified).
    radius_col, mass_col, density_col, flux_col, surface_temp_col : str
        Column names to use.

    Returns
    -------
    DataFrame
        Copy of ``df`` with the extra columns.
    """
    out = df.copy()

    def col(name: Optional[str]) -> pd.Series:
        if name and name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    r, m, rho_gcc = col(radius_col), col(mass_col), col(density_col)
    flux, temp = col(flux_col), col(surface_temp_col)

    dens_mr = density_from_mass_radius(m, r)
    dens_cat = rho_gcc.where(rho_gcc > 0) / EARTH_DENSITY_GCC
    density = dens_mr.fillna(dens_cat)

    vesc_mr = escape_velocity_from_mass_radius(m, r)
    with np.errstate(invalid="ignore"):
        vesc_rd = r.where(r > 0) * np.sqrt(dens_cat)
    vesc = vesc_mr.fillna(vesc_rd)

    out["density_eu"] = density.astype(float)
    out["escape_velocity_eu"] = vesc.astype(float)
    out["esi_2p"] = compute_esi_2p(r, flux)
    res = compute_esi_4p(r, out["density_eu"], out["escape_velocity_eu"], temp)
    out["esi_4p_interior"] = res.interior
    out["esi_4p_surface"] = res.surface
    out["esi_4p"] = res.esi
    return out


def _cache_path(cache_dir: Path, query: str) -> Path:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"pscomppars_{digest}.csv"


def _fetch(query: str, timeout: float, session: Optional[requests.Session]) -> str:
    getter = session.get if session is not None else requests.get
    resp = getter(TAP_URL, params={"query": query, "format": "csv"}, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    if text.lstrip().startswith("<"):
        # TAP errors are returned as VOTable/XML documents.
        raise RuntimeError(f"NASA Exoplanet Archive returned an error document:\n{text[:500]}")
    return text


def load_nasa_exoplanets(
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    refresh: bool = False,
    max_age_days: Optional[float] = None,
    timeout: float = 120.0,
    add_metrics: bool = True,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Load confirmed exoplanets from the NASA Exoplanet Archive.

    The first call downloads ``pscomppars`` through the TAP API (no API key)
    and writes the raw CSV to the cache; later calls with the same query read
    the cache and work offline. If a refresh fails because the network is
    unavailable, the cached copy is returned with a warning.

    Parameters
    ----------
    columns : sequence of str, optional
        Archive columns to select (default :data:`DEFAULT_COLUMNS`). Columns
        needed for the ESI are always included.
    where : str, optional
        ADQL filter, e.g. ``"pl_rade < 2"``.
    cache_dir : str or Path, optional
        Cache location (default :func:`default_cache_dir`).
    refresh : bool, default False
        Ignore the cache and re-download.
    max_age_days : float, optional
        Re-download if the cache file is older than this. ``None`` = never
        expire.
    timeout : float, default 120
        HTTP timeout in seconds.
    add_metrics : bool, default True
        Append derived columns via :func:`add_habitability_columns`.
    session : requests.Session, optional
        Session to use for the HTTP request.

    Returns
    -------
    DataFrame
        One row per planet.

    Raises
    ------
    requests.RequestException or RuntimeError
        If the download fails and no cached copy exists.
    """
    query = build_query(columns, where)
    cdir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    path = _cache_path(cdir, query)

    use_cache = path.exists() and not refresh
    if use_cache and max_age_days is not None:
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        use_cache = age_days <= max_age_days

    if use_cache:
        text = path.read_text(encoding="utf-8")
    else:
        try:
            text = _fetch(query, timeout, session)
        except (requests.RequestException, RuntimeError) as exc:
            if path.exists():
                warnings.warn(
                    f"Could not reach the NASA Exoplanet Archive ({exc}); using cached data "
                    f"from {path}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                text = path.read_text(encoding="utf-8")
            else:
                raise
        else:
            cdir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)

    df = pd.read_csv(io.StringIO(text))
    return add_habitability_columns(df) if add_metrics else df
