"""Circumstellar habitable-zone boundaries from Kopparapu et al. (2014).

The effective stellar flux at each boundary is a 4th-order polynomial fit::

    S_eff = S_eff_sun + a*T + b*T**2 + c*T**3 + d*T**4,   T = T_eff - 5780 K

and the corresponding orbital distance is ``d = sqrt(L / S_eff)`` AU with ``L``
in solar luminosities (Kopparapu et al. 2013, eq. 3).

Coefficients are copied verbatim from the authors' public calculator
``hzcalc.py`` (R. K. Kopparapu, 2014) for a 1 Earth-mass planet, plus the 5 and
0.1 Earth-mass runaway-greenhouse limits. The fits are valid for
2600 K <= T_eff <= 7200 K.

References
----------
Kopparapu, R. K. et al. (2014), ApJL 787, L29, arXiv:1404.5292.
Kopparapu, R. K. et al. (2013), ApJ 765, 131, arXiv:1301.6674.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ._utils import ArrayLike, restore, series_index, to_array

__all__ = [
    "KOPPARAPU_2014",
    "TEFF_RANGE",
    "kopparapu_seff",
    "hz_distance",
]

KOPPARAPU_2014: Dict[str, Tuple[float, float, float, float, float]] = {
    # limit: (S_eff_sun, a, b, c, d)
    "recent_venus": (1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15),
    "runaway_greenhouse": (1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15),
    "maximum_greenhouse": (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16),
    "early_mars": (0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16),
    "runaway_greenhouse_5me": (1.188, 1.433e-4, 1.707e-8, -8.968e-12, -2.084e-15),
    "runaway_greenhouse_0.1me": (0.99, 1.209e-4, 1.404e-8, -7.418e-12, -1.713e-15),
}
"""Kopparapu et al. (2014) HZ coefficients.

Conservative HZ: ``runaway_greenhouse`` (inner) to ``maximum_greenhouse`` (outer).
Optimistic HZ: ``recent_venus`` (inner) to ``early_mars`` (outer).
"""

TEFF_RANGE: Tuple[float, float] = (2600.0, 7200.0)
"""Stellar effective temperature range (K) over which the fits are valid."""


def kopparapu_seff(
    teff: ArrayLike,
    limit: str = "runaway_greenhouse",
    extrapolate: bool = False,
) -> Any:
    """Effective stellar flux at a habitable-zone boundary.

    Parameters
    ----------
    teff : array-like
        Stellar effective temperature in K.
    limit : str, default "runaway_greenhouse"
        One of the keys of :data:`KOPPARAPU_2014`.
    extrapolate : bool, default False
        If False, return NaN for ``teff`` outside 2600-7200 K, where the
        published fit is not valid.

    Returns
    -------
    float, ndarray or Series
        Effective flux ``S_eff`` in units of the present solar constant at Earth.
    """
    if limit not in KOPPARAPU_2014:
        raise KeyError(f"Unknown HZ limit '{limit}'. Choose from {sorted(KOPPARAPU_2014)}")
    s0, a, b, c, d = KOPPARAPU_2014[limit]
    idx = series_index(teff)
    t_eff = to_array(teff)
    t = t_eff - 5780.0
    seff = s0 + a * t + b * t**2 + c * t**3 + d * t**4
    valid = np.isfinite(t_eff)
    if not extrapolate:
        valid &= (t_eff >= TEFF_RANGE[0]) & (t_eff <= TEFF_RANGE[1])
    return restore(np.where(valid, seff, np.nan), idx, f"seff_{limit}")


def hz_distance(luminosity: ArrayLike, seff: ArrayLike, log_luminosity: bool = False) -> Any:
    """Orbital distance (AU) receiving effective flux ``seff``.

    ``d = sqrt(L / S_eff)`` (Kopparapu et al. 2013, eq. 3).

    Parameters
    ----------
    luminosity : array-like
        Stellar luminosity in solar units, or ``log10(L/L_sun)`` if
        ``log_luminosity`` is True (as in the NASA ``st_lum`` column).
    seff : array-like
        Effective flux from :func:`kopparapu_seff`.
    log_luminosity : bool, default False
        Interpret ``luminosity`` as log10.

    Returns
    -------
    float, ndarray or Series
        Distance in AU; NaN for non-positive luminosity or flux.
    """
    idx = series_index(luminosity, seff)
    lum = to_array(luminosity)
    if log_luminosity:
        lum = 10.0**lum
    s = to_array(seff)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(lum) & np.isfinite(s) & (lum > 0) & (s > 0)
        d = np.sqrt(np.where(valid, lum / np.where(valid, s, 1.0), np.nan))
    return restore(np.where(valid, d, np.nan), idx, "hz_distance_au")
