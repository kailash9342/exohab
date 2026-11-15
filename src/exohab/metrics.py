"""Planetary habitability and similarity metrics.

All functions are vectorised: they accept Python scalars, lists, NumPy arrays or
pandas Series (positional broadcasting). A scalar-only call returns a ``float``;
if any input is a Series the result is a Series carrying that Series' index;
otherwise an ``ndarray`` is returned.

Invalid physical inputs (negative values, NaN, ``None`` or infinities) yield NaN
rather than raising, so whole catalogues can be processed in one call.

References
----------
.. [SM11] Schulze-Makuch, D. et al. (2011). "A Two-Tiered Approach to Assessing
   the Habitability of Exoplanets." Astrobiology 11(10), 1041-1052.
   doi:10.1089/ast.2010.0592
.. [PHL] Méndez, A. Planetary Habitability Laboratory, University of Puerto Rico
   at Arecibo: "Earth Similarity Index (ESI)", https://phl.upr.edu
.. [K17] Kashyap Jagadeesh, M. (2017). "Earth Similarity Index and
   Habitability Studies of Exoplanets", M.Phil. dissertation, Christ University
   (arXiv:1801.07101), Sec. 3.2 and Table 3.2; companion paper: Kashyap
   Jagadeesh, Gudennavar, Doshi & Safonova, "Similarity indexing of exoplanets
   in search for potential habitability: application to Mars-like worlds",
   Astrophysics and Space Science (2017).
.. [I14] Irwin, L. N., Méndez, A., Fairén, A. G., Schulze-Makuch, D. (2014).
   "Assessing the Possibility of Biological Complexity on Other Worlds, with an
   Estimate of the Occurrence of Complex Life in the Milky Way Galaxy."
   Challenges 5(1), 159-174. doi:10.3390/challe5010159
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional, Union

import numpy as np

from ._utils import ArrayLike, restore, series_index, to_array

__all__ = [
    "EARTH_RADIUS_KM",
    "EARTH_DENSITY_GCC",
    "EARTH_ESCAPE_VELOCITY_KMS",
    "EARTH_SURFACE_TEMP_K",
    "ESI_WEIGHTS",
    "EARTH_REFERENCE",
    "REFERENCE_BODIES",
    "ESIResult",
    "density_from_mass_radius",
    "escape_velocity_from_mass_radius",
    "compute_esi_4p",
    "compute_esi_2p",
    "compute_similarity",
    "compute_phi",
    "compute_bci",
    "SURFACE_TEMP_METHODS",
    "estimate_surface_temp",
]

# ---------------------------------------------------------------------------
# Reference values (Schulze-Makuch et al. 2011, Table 1; PHL ESI definition)
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM: float = 6371.0
"""Earth mean radius in km (1 Earth unit, EU)."""
EARTH_DENSITY_GCC: float = 5.51
"""Earth bulk density in g/cm^3 (1 EU) as used in the ESI definition."""
EARTH_ESCAPE_VELOCITY_KMS: float = 11.19
"""Earth escape velocity in km/s (1 EU)."""
EARTH_SURFACE_TEMP_K: float = 288.0
"""Earth mean surface temperature in K (ESI reference value)."""

ESI_WEIGHTS: Dict[str, float] = {
    "radius": 0.57,
    "density": 1.07,
    "escape_velocity": 0.70,
    "surface_temp": 5.58,
}
"""ESI weight exponents (Schulze-Makuch et al. 2011, Table 1)."""

EARTH_REFERENCE: Dict[str, float] = {
    "radius": 1.0,
    "density": 1.0,
    "escape_velocity": 1.0,
    "surface_temp": EARTH_SURFACE_TEMP_K,
}

_PARAMS = ("radius", "density", "escape_velocity", "surface_temp")

REFERENCE_BODIES: Dict[str, Dict[str, Any]] = {
    "earth": {
        "radius": 1.0,
        "density": 1.0,
        "escape_velocity": 1.0,
        "surface_temp": 288.0,
        "weights": dict(ESI_WEIGHTS),
        "source": "Schulze-Makuch et al. (2011) ESI reference values and weights.",
    },
    "mars": {
        # Kashyap Jagadeesh et al. (2017) Mars Similarity Index (MSI):
        # R = 3390 km, rho = 3.93 g/cm^3, v_e = 5.03 km/s, T_s = 240 K,
        # weights 0.86, 2.10, 1.09, 3.23 (Table 3.2 of arXiv:1801.07101).
        "radius": 3390.0 / EARTH_RADIUS_KM,
        "density": 3.93 / EARTH_DENSITY_GCC,
        "escape_velocity": 5.03 / EARTH_ESCAPE_VELOCITY_KMS,
        "surface_temp": 240.0,
        "weights": {
            "radius": 0.86,
            "density": 2.10,
            "escape_velocity": 1.09,
            "surface_temp": 3.23,
        },
        "source": "Mars Similarity Index, Kashyap Jagadeesh et al. (2017).",
    },
    "venus": {
        # NASA GSFC Planetary Fact Sheet: R = 6051.8 km, rho = 5.243 g/cm^3,
        # v_e = 10.36 km/s, mean surface temperature 737 K.
        "radius": 6051.8 / EARTH_RADIUS_KM,
        "density": 5.243 / EARTH_DENSITY_GCC,
        "escape_velocity": 10.36 / EARTH_ESCAPE_VELOCITY_KMS,
        "surface_temp": 737.0,
        "weights": dict(ESI_WEIGHTS),
        "source": (
            "Reference values: NASA Planetary Fact Sheet. No body-specific weights "
            "are published; the Schulze-Makuch et al. (2011) ESI weights are reused."
        ),
    },
    "titan": {
        # NASA GSFC Saturnian Satellite Fact Sheet: R = 2574.7 km,
        # rho = 1.880 g/cm^3, v_e = 2.639 km/s, surface temperature ~94 K.
        "radius": 2574.7 / EARTH_RADIUS_KM,
        "density": 1.880 / EARTH_DENSITY_GCC,
        "escape_velocity": 2.639 / EARTH_ESCAPE_VELOCITY_KMS,
        "surface_temp": 94.0,
        "weights": dict(ESI_WEIGHTS),
        "source": (
            "Reference values: NASA Saturnian Satellite Fact Sheet. No body-specific "
            "weights are published; the Schulze-Makuch et al. (2011) ESI weights are reused."
        ),
    },
}
"""Built-in reference bodies for :func:`compute_similarity`.

Radius, density and escape velocity are stored in Earth units; surface
temperature in kelvin.
"""


class ESIResult(NamedTuple):
    """Interior, surface and global similarity index.

    Attributes
    ----------
    interior : float, ndarray or Series
        Interior index (radius and bulk density).
    surface : float, ndarray or Series
        Surface index (escape velocity and surface temperature).
    esi : float, ndarray or Series
        Global index, ``sqrt(interior * surface)``.
    """

    interior: Any
    surface: Any
    esi: Any


# ---------------------------------------------------------------------------
# Derived physical quantities
# ---------------------------------------------------------------------------
def density_from_mass_radius(mass: ArrayLike, radius: ArrayLike) -> Any:
    """Bulk density in Earth units from mass and radius in Earth units.

    ``rho / rho_E = (M / M_E) / (R / R_E)**3`` (uniform-sphere definition of
    bulk density).

    Parameters
    ----------
    mass : array-like
        Planet mass in Earth masses.
    radius : array-like
        Planet radius in Earth radii.

    Returns
    -------
    float, ndarray or Series
        Bulk density in Earth units; NaN where mass < 0 or radius <= 0.
    """
    idx = series_index(mass, radius)
    m, r = to_array(mass), to_array(radius)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(m) & np.isfinite(r) & (m >= 0) & (r > 0)
        out = np.where(valid, m / r**3, np.nan)
    return restore(out, idx, "density_eu")


def escape_velocity_from_mass_radius(mass: ArrayLike, radius: ArrayLike) -> Any:
    """Escape velocity in Earth units from mass and radius in Earth units.

    From ``v_e = sqrt(2 G M / R)``: ``v_e / v_e,E = sqrt((M / M_E) / (R / R_E))``.

    Parameters
    ----------
    mass : array-like
        Planet mass in Earth masses.
    radius : array-like
        Planet radius in Earth radii.

    Returns
    -------
    float, ndarray or Series
        Escape velocity in Earth units; NaN where mass < 0 or radius <= 0.
    """
    idx = series_index(mass, radius)
    m, r = to_array(mass), to_array(radius)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(m) & np.isfinite(r) & (m >= 0) & (r > 0)
        out = np.where(valid, np.sqrt(np.where(valid, m / r, np.nan)), np.nan)
    return restore(out, idx, "escape_velocity_eu")


# ---------------------------------------------------------------------------
# Surface temperature from equilibrium temperature
# ---------------------------------------------------------------------------
SURFACE_TEMP_METHODS: Dict[str, str] = {
    "kashyap2017": (
        "T_s = 9.650 + 1.096 * T_eq (K); least-squares fit to Solar System bodies "
        "with a rocky surface and an atmosphere, Kashyap Jagadeesh (2017), eq. 3.12."
    ),
}
"""Available methods for :func:`estimate_surface_temp`."""


def estimate_surface_temp(teq: ArrayLike, method: str = "kashyap2017") -> Any:
    """Estimate mean surface temperature from equilibrium temperature.

    Exoplanet catalogues list the equilibrium temperature ``T_eq``, but the
    4-parameter ESI is defined with the mean *surface* temperature (Earth:
    T_eq ~ 255 K, T_s = 288 K).

    ``method="kashyap2017"`` uses the linear relation [K17]_ (eq. 3.12)::

        T_s = 9.650 + 1.096 * T_eq

    fitted to Earth, Mars, the Moon, Titan and Triton (T_eq ~ 34-255 K);
    Venus, with its runaway greenhouse, was excluded from the fit. With the
    published (rounded) coefficients it gives 289.1 K for Earth
    (T_eq = 255 K) and 247.5 K for Mars (T_eq = 217 K); the source's Table 3.3
    lists 289.28 K and 247.61 K, the small difference coming from coefficient
    rounding.

    .. warning::
       The relation is empirical, rests on five bodies, and only applies to
       rocky planets with an atmosphere. It is not meaningful for gas giants,
       and is an extrapolation for T_eq above ~255 K.

    Parameters
    ----------
    teq : array-like
        Equilibrium temperature in K.
    method : str, default "kashyap2017"
        Key of :data:`SURFACE_TEMP_METHODS`.

    Returns
    -------
    float, ndarray or Series
        Estimated surface temperature in K; NaN for non-positive or missing
        ``teq``.
    """
    if method not in SURFACE_TEMP_METHODS:
        raise KeyError(f"Unknown method '{method}'. Choose from {sorted(SURFACE_TEMP_METHODS)}")
    idx = series_index(teq)
    t = to_array(teq)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(t) & (t > 0)
        ts = 9.650 + 1.096 * t
    return restore(np.where(valid, ts, np.nan), idx, "surface_temp_est")


# ---------------------------------------------------------------------------
# Earth Similarity Index
# ---------------------------------------------------------------------------
def _similarity_term(x: np.ndarray, x0: float, w: float) -> np.ndarray:
    """Single-property similarity ``(1 - |x - x0| / (x + x0)) ** w``.

    Returns NaN for non-finite or negative ``x``. ``x = 0`` gives 0.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(x) & (x >= 0)
        xs = np.where(valid, x, x0)  # placeholder avoids warnings on invalid entries
        term = (1.0 - np.abs(xs - x0) / (xs + x0)) ** w
    return np.where(valid, term, np.nan)


def _check_reference(ref: Mapping[str, float], weights: Mapping[str, float]) -> None:
    for key in _PARAMS:
        if key not in ref:
            raise KeyError(f"reference is missing '{key}'")
        if key not in weights:
            raise KeyError(f"weights is missing '{key}'")
        if not (np.isfinite(ref[key]) and ref[key] > 0):
            raise ValueError(f"reference value for '{key}' must be a positive finite number")
        if not (np.isfinite(weights[key]) and weights[key] > 0):
            raise ValueError(f"weight for '{key}' must be a positive finite number")


def _esi_core(
    radius: ArrayLike,
    density: ArrayLike,
    escape_velocity: ArrayLike,
    surface_temp: ArrayLike,
    ref: Mapping[str, float],
    weights: Mapping[str, float],
) -> ESIResult:
    _check_reference(ref, weights)
    idx = series_index(radius, density, escape_velocity, surface_temp)
    r, d, v, t = (to_array(a) for a in (radius, density, escape_velocity, surface_temp))
    s_r = _similarity_term(r, ref["radius"], weights["radius"])
    s_d = _similarity_term(d, ref["density"], weights["density"])
    s_v = _similarity_term(v, ref["escape_velocity"], weights["escape_velocity"])
    s_t = _similarity_term(t, ref["surface_temp"], weights["surface_temp"])
    interior = np.sqrt(s_r * s_d)
    surface = np.sqrt(s_v * s_t)
    glob = np.sqrt(interior * surface)
    return ESIResult(
        restore(interior, idx, "esi_interior"),
        restore(surface, idx, "esi_surface"),
        restore(glob, idx, "esi"),
    )


def compute_esi_4p(
    radius: ArrayLike,
    density: ArrayLike,
    escape_velocity: ArrayLike,
    surface_temp: ArrayLike,
    weights: Optional[Mapping[str, float]] = None,
) -> ESIResult:
    """Four-parameter Earth Similarity Index (Schulze-Makuch et al. 2011).

    For each property ``x`` with Earth reference ``x0`` and weight ``w``::

        ESI_x = (1 - |x - x0| / (x + x0)) ** w

    and the indices are combined as geometric means::

        ESI_I = sqrt(ESI_radius * ESI_density)               # interior
        ESI_S = sqrt(ESI_escape_velocity * ESI_temperature)  # surface
        ESI   = sqrt(ESI_I * ESI_S)                          # global

    This is algebraically identical to the PHL form
    ``ESI = prod_i (1 - |(x_i - x_i0)/(x_i + x_i0)|) ** (w_i / n)`` with ``n = 4``.

    Reference values / weights [SM11]_: radius 1 EU (6371 km), w = 0.57;
    bulk density 1 EU (5.51 g cm^-3), w = 1.07; escape velocity 1 EU
    (11.19 km s^-1), w = 0.70; mean surface temperature 288 K, w = 5.58.

    Parameters
    ----------
    radius : array-like
        Mean radius in Earth radii.
    density : array-like
        Bulk density in Earth units.
    escape_velocity : array-like
        Escape velocity in Earth units.
    surface_temp : array-like
        Mean *surface* temperature in kelvin. Catalogue equilibrium
        temperatures are not surface temperatures (Earth: T_eq ~ 255 K vs
        T_s = 288 K); using them as a proxy biases the surface ESI.
    weights : mapping, optional
        Override weight exponents (keys ``radius``, ``density``,
        ``escape_velocity``, ``surface_temp``). Defaults to :data:`ESI_WEIGHTS`.

    Returns
    -------
    ESIResult
        Named tuple ``(interior, surface, esi)``, each in [0, 1]; NaN where any
        required input is missing or negative.

    Examples
    --------
    >>> res = compute_esi_4p(0.53, 0.71, 0.45, 240.0)  # Mars, Kashyap et al. 2017
    >>> round(res.esi, 2)
    0.73
    """
    w = dict(ESI_WEIGHTS) if weights is None else {**ESI_WEIGHTS, **weights}
    return _esi_core(radius, density, escape_velocity, surface_temp, EARTH_REFERENCE, w)


def compute_esi_2p(radius: ArrayLike, flux: ArrayLike) -> Any:
    """Two-parameter (radius + stellar flux) Earth Similarity Index (PHL).

    Used by the Planetary Habitability Laboratory for transiting planets where
    only radius and insolation are known::

        ESI = 1 - sqrt( 0.5 * [ ((S - S_E)/(S + S_E))**2 + ((R - R_E)/(R + R_E))**2 ] )

    with ``S_E = 1`` (Earth insolation) and ``R_E = 1`` (Earth radius). This is
    one minus the quadratic mean of the two normalised distances; it is *not*
    numerically identical to the 4-parameter ESI. Worked check from
    arXiv:1712.01040: Proxima b (R = 1.12, S = 0.700522) gives ESI = 0.8692.

    Parameters
    ----------
    radius : array-like
        Planet radius in Earth radii (>= 0).
    flux : array-like
        Stellar insolation flux in Earth units (>= 0).

    Returns
    -------
    float, ndarray or Series
        ESI in [0, 1]; NaN for negative or missing inputs.
    """
    idx = series_index(radius, flux)
    r, s = to_array(radius), to_array(flux)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(r) & np.isfinite(s) & (r >= 0) & (s >= 0)
        rs = np.where(valid, r, 1.0)
        ss = np.where(valid, s, 1.0)
        d_r = (rs - 1.0) / (rs + 1.0)
        d_s = (ss - 1.0) / (ss + 1.0)
        esi = 1.0 - np.sqrt(0.5 * (d_s**2 + d_r**2))
    return restore(np.where(valid, esi, np.nan), idx, "esi_2p")


def compute_similarity(
    radius: ArrayLike,
    density: ArrayLike,
    escape_velocity: ArrayLike,
    surface_temp: ArrayLike,
    reference: Union[str, Mapping[str, Any]] = "earth",
    weights: Optional[Mapping[str, float]] = None,
) -> ESIResult:
    """Similarity of a planet to an arbitrary reference body.

    Applies exactly the ESI functional form of :func:`compute_esi_4p` but with
    the reference values ``x0`` (and optionally weights) of another body.

    Built-in presets (``REFERENCE_BODIES``):

    * ``"earth"`` - standard ESI (Schulze-Makuch et al. 2011).
    * ``"mars"`` - Mars Similarity Index (MSI) with its published reference
      values (3390 km, 3.93 g cm^-3, 5.03 km s^-1, 240 K) and weights
      (0.86, 2.10, 1.09, 3.23) from Kashyap Jagadeesh et al. (2017).
    * ``"venus"``, ``"titan"`` - reference values from the NASA planetary /
      satellite fact sheets. No body-specific weights have been published, so
      the Earth ESI weights are reused (documented convention, override via
      ``weights``).

    Parameters
    ----------
    radius, density, escape_velocity : array-like
        Planet properties in **Earth units** (also for non-Earth references).
    surface_temp : array-like
        Mean surface temperature in kelvin.
    reference : str or mapping, default "earth"
        Preset name (case-insensitive) or a mapping with keys ``radius``,
        ``density``, ``escape_velocity`` (Earth units), ``surface_temp`` (K)
        and optionally ``weights``.
    weights : mapping, optional
        Weight exponents overriding those of the reference.

    Returns
    -------
    ESIResult
        ``(interior, surface, esi)`` similarity to the reference body.

    Raises
    ------
    KeyError
        Unknown preset or missing keys in a custom reference.
    ValueError
        Non-positive reference values or weights.
    """
    if isinstance(reference, str):
        key = reference.strip().lower()
        if key not in REFERENCE_BODIES:
            raise KeyError(
                f"Unknown reference body '{reference}'. "
                f"Available presets: {sorted(REFERENCE_BODIES)}"
            )
        ref: Dict[str, Any] = REFERENCE_BODIES[key]
    else:
        ref = dict(reference)
    base_w = dict(ref.get("weights") or ESI_WEIGHTS)
    if weights is not None:
        base_w.update(weights)
    ref_vals = {k: float(ref[k]) for k in _PARAMS if k in ref}
    return _esi_core(radius, density, escape_velocity, surface_temp, ref_vals, base_w)


# ---------------------------------------------------------------------------
# PHI and BCI
# ---------------------------------------------------------------------------
def _unit_interval(*arrays: np.ndarray) -> np.ndarray:
    valid = np.ones(np.broadcast(*arrays).shape, dtype=bool)
    for a in arrays:
        valid &= np.isfinite(a) & (a >= 0) & (a <= 1)
    return valid


def compute_phi(
    substrate: ArrayLike,
    energy: ArrayLike,
    chemistry: ArrayLike,
    liquid: ArrayLike,
) -> Any:
    """Planetary Habitability Index (Schulze-Makuch et al. 2011).

    ``PHI = (S * E * C * L) ** (1/4)``

    the geometric mean of four factor scores for a stable substrate (S),
    available energy (E), appropriate polymeric chemistry (C) and a liquid
    solvent (L). Each factor must already be scored on [0, 1] following the
    categorical rubric of [SM11]_ - this function performs the combination
    only and does not invent a scoring of raw physical parameters.

    Parameters
    ----------
    substrate, energy, chemistry, liquid : array-like
        Factor scores in [0, 1].

    Returns
    -------
    float, ndarray or Series
        PHI in [0, 1]; NaN where any factor is outside [0, 1] or missing.
    """
    idx = series_index(substrate, energy, chemistry, liquid)
    arrs = [to_array(a) for a in (substrate, energy, chemistry, liquid)]
    valid = _unit_interval(*arrs)
    with np.errstate(invalid="ignore"):
        phi = np.prod(np.broadcast_arrays(*arrs), axis=0) ** 0.25
    return restore(np.where(valid, phi, np.nan), idx, "phi")


def compute_bci(
    substrate: ArrayLike,
    energy: ArrayLike,
    temperature: ArrayLike,
    geophysics: ArrayLike,
    age: ArrayLike,
    relative: bool = False,
) -> Any:
    """Biological Complexity Index (Irwin et al. 2014).

    ``BCI_abs = (S * E * T * G * A) ** (1/5)``

    the geometric mean of factor scores for substrate (S), energy (E),
    temperature (T), geophysical activity (G) and age (A). The relative index
    is ``BCI_rel = BCI_abs / max(BCI_abs)`` over the evaluated set [I14]_.

    Each factor must be pre-scored on [0, 1] using the rubric of [I14]_; this
    function implements only the published combination step.

    Parameters
    ----------
    substrate, energy, temperature, geophysics, age : array-like
        Factor scores in [0, 1].
    relative : bool, default False
        If True return ``BCI_rel`` (normalised by the maximum finite BCI_abs
        among the inputs) instead of ``BCI_abs``.

    Returns
    -------
    float, ndarray or Series
        BCI in [0, 1]; NaN where any factor is outside [0, 1] or missing.
    """
    idx = series_index(substrate, energy, temperature, geophysics, age)
    arrs = [to_array(a) for a in (substrate, energy, temperature, geophysics, age)]
    valid = _unit_interval(*arrs)
    with np.errstate(invalid="ignore"):
        bci = np.prod(np.broadcast_arrays(*arrs), axis=0) ** 0.2
    bci = np.where(valid, bci, np.nan)
    if relative:
        finite = bci[np.isfinite(bci)]
        if finite.size == 0 or np.max(finite) == 0:
            bci = np.full_like(bci, np.nan)
        else:
            bci = bci / np.max(finite)
    return restore(bci, idx, "bci_rel" if relative else "bci_abs")
