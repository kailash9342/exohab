"""Monte Carlo propagation of (asymmetric) measurement errors into the ESI.

Each input may be

* a float - treated as exact (no uncertainty);
* ``(nominal, err)`` - symmetric 1-sigma error;
* ``(nominal, plus_err, minus_err)`` - asymmetric 1-sigma errors. The sign of
  ``minus_err`` is ignored, so NASA Exoplanet Archive ``err1``/``err2`` pairs
  (e.g. ``+0.11``/``-0.18``) can be passed directly.

Asymmetric errors are sampled from the two-piece ("split") normal
distribution, the standard interpretation of ``x^{+a}_{-b}`` quotes: a normal
with sigma ``b`` below the mode and sigma ``a`` above, joined continuously at
the mode. Draws fall below the mode with probability ``b / (a + b)``; the mean
is ``mode + sqrt(2/pi) * (a - b)`` (Wallis 2014, "The two-piece normal, binormal,
or double Gaussian distribution", Statistical Science 29, 106).

Parameters are sampled independently unless ``mass`` is supplied to
:func:`compute_esi_mc`, in which case density and escape velocity are derived
per draw from the same (radius, mass) pair so their correlation is preserved.

Non-physical draws (radius, mass, flux, temperature <= 0) are handled in one of
two ways:

* ``truncate=False`` (default): they yield NaN in the metric and are discarded;
  ``n_valid`` reports how many remain.
* ``truncate=True``: they are redrawn (rejection sampling), i.e. each input
  follows the split normal truncated at zero and ``n_valid == n_samples``.

Both correspond to the same truncated input distribution; ``truncate=True``
just keeps the requested sample size.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import (
    ESI_WEIGHTS,
    compute_esi_2p,
    compute_esi_4p,
    density_from_mass_radius,
    escape_velocity_from_mass_radius,
    estimate_surface_temp,
)

__all__ = [
    "MCResult",
    "parse_measurement",
    "sample_split_normal",
    "split_normal_cdf",
    "compute_esi_mc",
    "compute_esi_2p_mc",
    "compute_esi_mc_catalog",
]

Measurement = Union[float, int, Sequence[float]]
SeedLike = Optional[Union[int, np.random.Generator]]

_MIN_ACCEPTANCE = 1e-3
_MAX_REJECTION_ROUNDS = 200


@dataclass
class MCResult:
    """Summary of a Monte Carlo distribution.

    Attributes
    ----------
    mean, median, std : float
        Sample mean, median and standard deviation (ddof=1).
    ci68 : tuple of float
        Central 68.27% interval (15.865th - 84.135th percentiles).
    ci95 : tuple of float
        Central 95% interval (2.5th - 97.5th percentiles). Use
        :meth:`interval` for other levels.
    samples : ndarray
        Valid (finite) samples.
    n_valid : int
        Number of valid samples.
    n_samples : int
        Number of draws requested.
    components : dict
        Sub-results (e.g. ``"interior"``, ``"surface"`` for the 4-parameter ESI).
    """

    mean: float
    median: float
    std: float
    ci68: Tuple[float, float]
    ci95: Tuple[float, float]
    samples: np.ndarray = field(repr=False)
    n_valid: int
    n_samples: int
    components: Dict[str, "MCResult"] = field(default_factory=dict, repr=False)

    @classmethod
    def from_samples(cls, samples: np.ndarray, n_samples: Optional[int] = None) -> "MCResult":
        """Build a summary from raw draws (non-finite values are dropped)."""
        raw = np.asarray(samples, dtype=float).ravel()
        s = raw[np.isfinite(raw)]
        n_req = int(raw.size if n_samples is None else n_samples)
        if s.size == 0:
            nan = float("nan")
            return cls(nan, nan, nan, (nan, nan), (nan, nan), s, 0, n_req)
        lo68, hi68, lo95, hi95 = np.percentile(s, [15.865, 84.135, 2.5, 97.5])
        std = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
        return cls(
            mean=float(np.mean(s)),
            median=float(np.median(s)),
            std=std,
            ci68=(float(lo68), float(hi68)),
            ci95=(float(lo95), float(hi95)),
            samples=s,
            n_valid=int(s.size),
            n_samples=n_req,
        )

    def interval(self, level: float) -> Tuple[float, float]:
        """Central credible interval containing fraction ``level`` of samples."""
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        if self.n_valid == 0:
            return (float("nan"), float("nan"))
        tail = (1 - level) / 2 * 100
        lo, hi = np.percentile(self.samples, [tail, 100 - tail])
        return float(lo), float(hi)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict summary (without the sample array)."""
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "ci68_low": self.ci68[0],
            "ci68_high": self.ci68[1],
            "ci95_low": self.ci95[0],
            "ci95_high": self.ci95[1],
            "n_valid": self.n_valid,
            "n_samples": self.n_samples,
        }


def parse_measurement(value: Measurement) -> Tuple[float, float, float]:
    """Normalise a measurement to ``(nominal, plus_err, minus_err)``.

    Parameters
    ----------
    value : float or sequence
        ``x``, ``(x, err)`` or ``(x, plus_err, minus_err)``.

    Returns
    -------
    tuple of float
        Nominal value and non-negative upper / lower 1-sigma errors.

    Raises
    ------
    ValueError
        Wrong tuple length or non-finite values.
    """
    if np.isscalar(value):
        nom, plus, minus = float(value), 0.0, 0.0  # type: ignore[arg-type]
    else:
        vals = [float(v) for v in value]  # type: ignore[union-attr]
        if len(vals) == 2:
            nom, plus, minus = vals[0], abs(vals[1]), abs(vals[1])
        elif len(vals) == 3:
            nom, plus, minus = vals[0], abs(vals[1]), abs(vals[2])
        else:
            raise ValueError("measurement must be x, (x, err) or (x, plus_err, minus_err)")
    if not all(np.isfinite([nom, plus, minus])):
        raise ValueError(f"measurement contains non-finite values: {value!r}")
    return nom, plus, minus


def split_normal_cdf(x: Any, nominal: float, plus_err: float, minus_err: float) -> Any:
    """Cumulative distribution function of the two-piece normal.

    With ``s1 = minus_err``, ``s2 = plus_err`` and ``Phi`` the standard normal CDF::

        F(x) = 2 s1 / (s1 + s2) * Phi((x - mu) / s1)                    x <  mu
        F(x) = (s1 + s2 * (2 Phi((x - mu) / s2) - 1)) / (s1 + s2)      x >= mu

    Parameters
    ----------
    x : float or array-like
        Evaluation points.
    nominal, plus_err, minus_err : float
        Mode and 1-sigma widths.

    Returns
    -------
    float or ndarray
        ``P(X <= x)``.
    """
    s1, s2 = abs(float(minus_err)), abs(float(plus_err))
    xa = np.asarray(x, dtype=float)
    if s1 == 0 and s2 == 0:
        return np.where(xa >= nominal, 1.0, 0.0) if xa.ndim else float(xa >= nominal)
    phi = np.vectorize(lambda z: 0.5 * math.erfc(-z / math.sqrt(2.0)), otypes=[float])
    with np.errstate(divide="ignore", invalid="ignore"):
        z_lo = (xa - nominal) / s1 if s1 > 0 else np.where(xa < nominal, -np.inf, 0.0)
        z_hi = (xa - nominal) / s2 if s2 > 0 else np.where(xa >= nominal, np.inf, 0.0)
    below = 2 * s1 / (s1 + s2) * phi(z_lo)
    above = (s1 + s2 * (2 * phi(z_hi) - 1)) / (s1 + s2)
    out = np.where(xa < nominal, below, above)
    return float(out) if out.ndim == 0 else out


def _rng(seed: SeedLike) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _draw(nom: Any, plus: Any, minus: Any, shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Broadcasting split-normal sampler (no truncation)."""
    nom = np.asarray(nom, dtype=float)
    s_hi = np.abs(np.asarray(plus, dtype=float))
    s_lo = np.abs(np.asarray(minus, dtype=float))
    tot = s_hi + s_lo
    with np.errstate(invalid="ignore", divide="ignore"):
        p_below = np.where(tot > 0, s_lo / np.where(tot > 0, tot, 1.0), 0.0)
    z = np.abs(rng.standard_normal(shape))
    below = rng.random(shape) < p_below
    return np.where(below, nom - z * s_lo, nom + z * s_hi)


def sample_split_normal(
    nominal: float,
    plus_err: float,
    minus_err: float,
    size: int,
    rng: Optional[np.random.Generator] = None,
    lower: Optional[float] = None,
) -> np.ndarray:
    """Draw from a two-piece (split) normal distribution.

    Parameters
    ----------
    nominal : float
        Mode of the distribution (the quoted value).
    plus_err, minus_err : float
        Upper and lower 1-sigma widths (absolute values are used).
    size : int
        Number of draws.
    rng : numpy.random.Generator, optional
        Random generator; a fresh default generator is used if omitted.
    lower : float, optional
        If given, sample the distribution truncated to ``x > lower`` by
        rejection (draws ``<= lower`` are redrawn).

    Returns
    -------
    ndarray
        ``size`` samples.

    Raises
    ------
    ValueError
        If ``lower`` excludes more than 99.9% of the probability mass (the
        measurement is inconsistent with the bound).
    """
    rng = np.random.default_rng() if rng is None else rng
    s_hi, s_lo = abs(float(plus_err)), abs(float(minus_err))
    if lower is not None:
        p_accept = 1.0 - split_normal_cdf(lower, nominal, s_hi, s_lo)
        if p_accept < _MIN_ACCEPTANCE:
            raise ValueError(
                f"truncation at {lower} keeps only {p_accept:.2e} of the distribution "
                f"{nominal} +{s_hi}/-{s_lo}; the measurement is inconsistent with the bound"
            )
    if s_hi == 0 and s_lo == 0:
        return np.full(size, float(nominal))
    out = _draw(nominal, s_hi, s_lo, (size,), rng)
    if lower is not None:
        bad = out <= lower
        while bad.any():
            out[bad] = _draw(nominal, s_hi, s_lo, (int(bad.sum()),), rng)
            bad = out <= lower
    return out


def _check_n(n_samples: int) -> int:
    n = int(n_samples)
    if n < 2:
        raise ValueError("n_samples must be >= 2")
    return n


def _sample(m: Measurement, n: int, rng: np.random.Generator, truncate: bool) -> np.ndarray:
    return sample_split_normal(*parse_measurement(m), n, rng, lower=0.0 if truncate else None)


def compute_esi_mc(
    radius: Measurement,
    density: Optional[Measurement] = None,
    escape_velocity: Optional[Measurement] = None,
    surface_temp: Optional[Measurement] = None,
    *,
    mass: Optional[Measurement] = None,
    n_samples: int = 10_000,
    seed: SeedLike = None,
    weights: Optional[Mapping[str, float]] = None,
    truncate: bool = False,
) -> MCResult:
    """Monte Carlo distribution of the 4-parameter ESI.

    Either give ``density`` and ``escape_velocity`` directly, or give ``mass``
    (Earth masses) and leave them as None; then per draw
    ``density = M / R**3`` and ``escape_velocity = sqrt(M / R)`` (Earth units),
    which keeps their physical correlation through the shared radius and mass.

    Parameters
    ----------
    radius : float or tuple
        Radius in Earth radii (see module docstring for tuple formats).
    density, escape_velocity : float or tuple, optional
        In Earth units. Required unless ``mass`` is given.
    surface_temp : float or tuple
        Mean surface temperature in K.
    mass : float or tuple, optional
        Mass in Earth masses.
    n_samples : int, default 10000
        Number of Monte Carlo draws.
    seed : int or Generator, optional
        Seed for reproducibility.
    weights : mapping, optional
        ESI weight overrides (see :func:`exohab.compute_esi_4p`).
    truncate : bool, default False
        Redraw non-positive input draws instead of discarding them.

    Returns
    -------
    MCResult
        Global-ESI summary; ``components["interior"]`` and
        ``components["surface"]`` hold the sub-index distributions.
    """
    n = _check_n(n_samples)
    if surface_temp is None:
        raise ValueError("surface_temp is required")
    derive = density is None and escape_velocity is None
    if derive and mass is None:
        raise ValueError("provide density and escape_velocity, or mass to derive them")
    if not derive and (density is None or escape_velocity is None):
        raise ValueError("density and escape_velocity must both be given (or both omitted with mass)")
    if not derive and mass is not None:
        raise ValueError("give either mass or (density, escape_velocity), not both")

    rng = _rng(seed)
    r = _sample(radius, n, rng, truncate)
    t = _sample(surface_temp, n, rng, truncate)
    if derive:
        m = _sample(mass, n, rng, truncate)  # type: ignore[arg-type]
        d = density_from_mass_radius(m, r)
        v = escape_velocity_from_mass_radius(m, r)
    else:
        d = _sample(density, n, rng, truncate)  # type: ignore[arg-type]
        v = _sample(escape_velocity, n, rng, truncate)  # type: ignore[arg-type]

    w = dict(ESI_WEIGHTS) if weights is None else {**ESI_WEIGHTS, **weights}
    res = compute_esi_4p(r, d, v, t, weights=w)
    out = MCResult.from_samples(res.esi, n)
    out.components = {
        "interior": MCResult.from_samples(res.interior, n),
        "surface": MCResult.from_samples(res.surface, n),
    }
    return out


def compute_esi_2p_mc(
    radius: Measurement,
    flux: Measurement,
    *,
    n_samples: int = 10_000,
    seed: SeedLike = None,
    truncate: bool = False,
) -> MCResult:
    """Monte Carlo distribution of the 2-parameter (radius + flux) ESI.

    Parameters
    ----------
    radius : float or tuple
        Radius in Earth radii, e.g. ``(1.34, 0.11, -0.18)``.
    flux : float or tuple
        Insolation in Earth units.
    n_samples : int, default 10000
        Number of draws.
    seed : int or Generator, optional
        Seed for reproducibility.
    truncate : bool, default False
        Redraw non-positive input draws instead of discarding them.

    Returns
    -------
    MCResult
        Summary of the ESI distribution.
    """
    n = _check_n(n_samples)
    rng = _rng(seed)
    r = _sample(radius, n, rng, truncate)
    s = _sample(flux, n, rng, truncate)
    return MCResult.from_samples(compute_esi_2p(r, s), n)


# ---------------------------------------------------------------------------
# Whole-catalogue Monte Carlo
# ---------------------------------------------------------------------------
def _catalog_param(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nominal, +err, -err arrays for ``col`` (NASA ``err1``/``err2`` convention)."""
    if col not in df.columns:
        raise KeyError(f"column '{col}' not found")
    nom = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    e1 = df.get(f"{col}err1")
    e2 = df.get(f"{col}err2")
    plus = (
        np.abs(pd.to_numeric(e1, errors="coerce").to_numpy(dtype=float))
        if e1 is not None
        else np.full(len(df), np.nan)
    )
    minus = (
        np.abs(pd.to_numeric(e2, errors="coerce").to_numpy(dtype=float))
        if e2 is not None
        else np.full(len(df), np.nan)
    )
    complete = np.isfinite(plus) & np.isfinite(minus)
    return nom, np.nan_to_num(plus), np.nan_to_num(minus), complete


def _draw_matrix(
    nom: np.ndarray, plus: np.ndarray, minus: np.ndarray, n: int,
    rng: np.random.Generator, truncate: bool,
) -> np.ndarray:
    k = nom.shape[0]
    out = _draw(nom[:, None], plus[:, None], minus[:, None], (k, n), rng)
    if truncate:
        for _ in range(_MAX_REJECTION_ROUNDS):
            bad = out <= 0
            if not bad.any():
                break
            rows = np.nonzero(bad)[0]
            out[bad] = _draw(nom[rows], plus[rows], minus[rows], (rows.size,), rng)
        out[out <= 0] = np.nan  # hopeless rows (bound excludes almost all mass)
    return out


def compute_esi_mc_catalog(
    df: pd.DataFrame,
    method: str = "2p",
    n_samples: int = 2000,
    seed: SeedLike = None,
    truncate: bool = False,
    radius_col: str = "pl_rade",
    flux_col: str = "pl_insol",
    mass_col: str = "pl_bmasse",
    temp_col: str = "pl_eqt",
    surface_temp_method: Optional[str] = None,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Monte Carlo ESI distribution summaries for every row of a catalogue.

    Uses the NASA Exoplanet Archive error convention: for a column ``x`` the
    1-sigma errors are in ``xerr1`` (upper) and ``xerr2`` (lower, negative).
    Each planet's inputs are sampled from split normals (see module docstring).

    * ``method="2p"`` - radius and insolation (:func:`exohab.compute_esi_2p`).
    * ``method="4p"`` - radius, mass and temperature; density and escape
      velocity are derived per draw from mass and radius (correlated).
      ``temp_col`` is used as surface temperature unless
      ``surface_temp_method`` is given (then it is converted per draw with
      :func:`exohab.estimate_surface_temp`).

    Rows with a missing nominal value get NaN summaries. A missing error is
    treated as zero (exact value) and flagged by ``*_errors_complete = False``,
    so such intervals are lower limits on the true uncertainty.

    Parameters
    ----------
    df : DataFrame
        Catalogue, e.g. from :func:`exohab.load_nasa_exoplanets`.
    method : {"2p", "4p"}, default "2p"
        Which ESI to propagate.
    n_samples : int, default 2000
        Draws per planet.
    seed : int or Generator, optional
        Seed for reproducibility.
    truncate : bool, default False
        Redraw non-positive input draws instead of discarding them.
    radius_col, flux_col, mass_col, temp_col : str
        Column names.
    surface_temp_method : str, optional
        e.g. ``"kashyap2017"`` (4p only).
    chunk_size : int, default 500
        Rows processed per vectorised batch (bounds memory use).

    Returns
    -------
    DataFrame
        Same index as ``df`` with columns ``esi_{method}_mc_mean``, ``_median``,
        ``_std``, ``_ci68_low``, ``_ci68_high``, ``_ci95_low``, ``_ci95_high``,
        ``_n_valid`` and ``_errors_complete``.
    """
    if method not in ("2p", "4p"):
        raise ValueError("method must be '2p' or '4p'")
    n = _check_n(n_samples)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    rng = _rng(seed)

    cols = [radius_col, flux_col] if method == "2p" else [radius_col, mass_col, temp_col]
    params = [_catalog_param(df, c) for c in cols]
    complete = np.logical_and.reduce([p[3] for p in params])

    k = len(df)
    stats = {s: np.full(k, np.nan) for s in
             ("mean", "median", "std", "ci68_low", "ci68_high", "ci95_low", "ci95_high")}
    n_valid = np.zeros(k, dtype=int)

    for start in range(0, k, chunk_size):
        sl = slice(start, min(start + chunk_size, k))
        draws = [_draw_matrix(p[0][sl], p[1][sl], p[2][sl], n, rng, truncate) for p in params]
        if method == "2p":
            esi = compute_esi_2p(draws[0], draws[1])
        else:
            r, m, t = draws
            if surface_temp_method is not None:
                t = estimate_surface_temp(t, surface_temp_method)
            esi = compute_esi_4p(
                r, density_from_mass_radius(m, r), escape_velocity_from_mass_radius(m, r), t
            ).esi
        esi = np.asarray(esi, dtype=float)
        valid = np.isfinite(esi)
        n_valid[sl] = valid.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stats["mean"][sl] = np.nanmean(esi, axis=1)
            stats["median"][sl] = np.nanmedian(esi, axis=1)
            stats["std"][sl] = np.nanstd(esi, axis=1, ddof=1)
            q = np.nanpercentile(esi, [15.865, 84.135, 2.5, 97.5], axis=1)
        stats["ci68_low"][sl], stats["ci68_high"][sl] = q[0], q[1]
        stats["ci95_low"][sl], stats["ci95_high"][sl] = q[2], q[3]

    prefix = f"esi_{method}_mc_"
    out = pd.DataFrame({prefix + s: v for s, v in stats.items()}, index=df.index)
    zero = n_valid == 0
    out.loc[zero, [prefix + s for s in stats]] = np.nan
    out[prefix + "n_valid"] = n_valid
    out[prefix + "errors_complete"] = complete
    return out
