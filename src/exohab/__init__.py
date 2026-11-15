"""exohab - exoplanet habitability toolkit.

Core functions (numpy/pandas/scipy/requests only) are imported eagerly. The
optional ``ml`` and ``viz`` helpers are loaded lazily on first access, so
``import exohab`` works without scikit-learn or matplotlib installed.
"""

from __future__ import annotations

from typing import Any

from .data import add_habitability_columns, load_nasa_exoplanets
from .habitable_zone import KOPPARAPU_2014, hz_distance, kopparapu_seff
from .metrics import (
    ESI_WEIGHTS,
    REFERENCE_BODIES,
    ESIResult,
    compute_bci,
    compute_esi_2p,
    compute_esi_4p,
    compute_phi,
    compute_similarity,
    density_from_mass_radius,
    escape_velocity_from_mass_radius,
)
from .uncertainty import MCResult, compute_esi_2p_mc, compute_esi_mc, sample_split_normal

__version__ = "0.1.0"

_LAZY = {
    "HabitabilityFeatureTransformer": "ml",
    "plot_esi_scatter": "viz",
    "plot_mc_distribution": "viz",
}

__all__ = [
    "__version__",
    "ESIResult",
    "ESI_WEIGHTS",
    "REFERENCE_BODIES",
    "KOPPARAPU_2014",
    "MCResult",
    "compute_esi_4p",
    "compute_esi_2p",
    "compute_similarity",
    "compute_phi",
    "compute_bci",
    "density_from_mass_radius",
    "escape_velocity_from_mass_radius",
    "kopparapu_seff",
    "hz_distance",
    "compute_esi_mc",
    "compute_esi_2p_mc",
    "sample_split_normal",
    "load_nasa_exoplanets",
    "add_habitability_columns",
    *_LAZY,
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'exohab' has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY))
