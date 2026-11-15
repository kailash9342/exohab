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

Draws that are non-physical (negative radius, mass, flux, temperature...) yield
NaN in the metric and are discarded; ``n_valid`` reports how many remain. This
is equivalent to truncating the input distributions at zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .metrics import (
    ESI_WEIGHTS,
    compute_esi_2p,
    compute_esi_4p,
    density_from_mass_radius,
    escape_velocity_from_mass_radius,
)

__all__ = [
    "MCResult",
    "parse_measurement",
    "sample_split_normal",
    "compute_esi_mc",
    "compute_esi_2p_mc",
]

Measurement = Union[float, int, Sequence[float]]


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


def sample_split_normal(
    nominal: float,
    plus_err: float,
    minus_err: float,
    size: int,
    rng: Optional[np.random.Generator] = None,
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

    Returns
    -------
    ndarray
        ``size`` samples.
    """
    rng = np.random.default_rng() if rng is None else rng
    s_hi, s_lo = abs(float(plus_err)), abs(float(minus_err))
    if s_hi == 0 and s_lo == 0:
        return np.full(size, float(nominal))
    z = np.abs(rng.standard_normal(size))
    below = rng.random(size) < s_lo / (s_lo + s_hi)
    return np.where(below, nominal - z * s_lo, nominal + z * s_hi)


def _check_n(n_samples: int) -> int:
    n = int(n_samples)
    if n < 2:
        raise ValueError("n_samples must be >= 2")
    return n


def compute_esi_mc(
    radius: Measurement,
    density: Optional[Measurement] = None,
    escape_velocity: Optional[Measurement] = None,
    surface_temp: Optional[Measurement] = None,
    *,
    mass: Optional[Measurement] = None,
    n_samples: int = 10_000,
    seed: Optional[Union[int, np.random.Generator]] = None,
    weights: Optional[Mapping[str, float]] = None,
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

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    r = sample_split_normal(*parse_measurement(radius), n, rng)
    t = sample_split_normal(*parse_measurement(surface_temp), n, rng)
    if derive:
        m = sample_split_normal(*parse_measurement(mass), n, rng)  # type: ignore[arg-type]
        d = density_from_mass_radius(m, r)
        v = escape_velocity_from_mass_radius(m, r)
    else:
        d = sample_split_normal(*parse_measurement(density), n, rng)  # type: ignore[arg-type]
        v = sample_split_normal(*parse_measurement(escape_velocity), n, rng)  # type: ignore[arg-type]

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
    seed: Optional[Union[int, np.random.Generator]] = None,
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

    Returns
    -------
    MCResult
        Summary of the ESI distribution.
    """
    n = _check_n(n_samples)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    r = sample_split_normal(*parse_measurement(radius), n, rng)
    s = sample_split_normal(*parse_measurement(flux), n, rng)
    return MCResult.from_samples(compute_esi_2p(r, s), n)
