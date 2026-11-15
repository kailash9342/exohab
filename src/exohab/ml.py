"""scikit-learn compatible habitability feature engineering.

Requires the optional dependency ``scikit-learn``
(``pip install exohab[ml]`` or ``exohab[all]``). Importing this module without
it does not fail; instantiating the transformer raises a clear ImportError.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .data import add_habitability_columns
from .habitable_zone import hz_distance, kopparapu_seff

try:  # pragma: no cover - exercised implicitly depending on environment
    from sklearn.base import BaseEstimator, TransformerMixin

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover

    class BaseEstimator:  # type: ignore[no-redef]
        """Placeholder used when scikit-learn is not installed."""

    class TransformerMixin:  # type: ignore[no-redef]
        """Placeholder used when scikit-learn is not installed."""

    _HAS_SKLEARN = False

__all__ = ["HabitabilityFeatureTransformer", "FEATURE_NAMES"]

FEATURE_NAMES: List[str] = [
    "density_eu",
    "escape_velocity_eu",
    "esi_2p",
    "esi_4p_interior",
    "esi_4p_surface",
    "esi_4p",
    "esi_interior_surface_ratio",
    "hz_seff_inner",
    "hz_seff_outer",
    "hz_seff_inner_optimistic",
    "hz_seff_outer_optimistic",
    "hz_inner_au",
    "hz_outer_au",
    "in_conservative_hz",
    "in_optimistic_hz",
]


def _require_sklearn() -> None:
    if not _HAS_SKLEARN:
        raise ImportError(
            "HabitabilityFeatureTransformer requires scikit-learn. "
            "Install it with: pip install exohab[all]"
        )


class HabitabilityFeatureTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Engineer habitability features from raw planetary/stellar columns.

    Features produced (see :data:`FEATURE_NAMES`):

    * ``density_eu``, ``escape_velocity_eu`` - from mass and radius.
    * ``esi_2p``, ``esi_4p_interior``, ``esi_4p_surface``, ``esi_4p`` - see
      :func:`exohab.data.add_habitability_columns` (equilibrium temperature is
      used as a surface-temperature proxy unless another column is given).
    * ``esi_interior_surface_ratio`` - interior / surface ESI.
    * ``hz_seff_inner`` / ``hz_seff_outer`` - conservative HZ flux limits
      (runaway / maximum greenhouse) and ``*_optimistic`` (recent Venus / early
      Mars), Kopparapu et al. (2014), valid for 2600-7200 K (NaN outside).
    * ``hz_inner_au`` / ``hz_outer_au`` - conservative HZ distances,
      ``sqrt(L / S_eff)``.
    * ``in_conservative_hz`` / ``in_optimistic_hz`` - 1.0 if the planet's
      insolation lies within the flux limits, 0.0 if not, NaN if unknown.

    Parameters
    ----------
    radius_col, mass_col, density_col, flux_col, surface_temp_col, teff_col, luminosity_col : str
        Input column names (NASA Exoplanet Archive defaults).
    luminosity_is_log : bool, default True
        ``luminosity_col`` holds log10(L/L_sun) (as ``st_lum`` does).
    keep_original : bool, default False
        If True, return the input columns followed by the new features.
    surface_temp_method : str, optional
        e.g. ``"kashyap2017"`` to convert equilibrium to surface temperature
        before computing the 4-parameter ESI (see
        :func:`exohab.estimate_surface_temp`).

    Notes
    -----
    The transformer is stateless: ``fit`` only validates the input and
    records column names. Input must be a pandas DataFrame.
    """

    def __init__(
        self,
        radius_col: str = "pl_rade",
        mass_col: str = "pl_bmasse",
        density_col: Optional[str] = "pl_dens",
        flux_col: str = "pl_insol",
        surface_temp_col: str = "pl_eqt",
        teff_col: str = "st_teff",
        luminosity_col: str = "st_lum",
        luminosity_is_log: bool = True,
        keep_original: bool = False,
        surface_temp_method: Optional[str] = None,
    ) -> None:
        _require_sklearn()
        self.radius_col = radius_col
        self.mass_col = mass_col
        self.density_col = density_col
        self.flux_col = flux_col
        self.surface_temp_col = surface_temp_col
        self.teff_col = teff_col
        self.luminosity_col = luminosity_col
        self.luminosity_is_log = luminosity_is_log
        self.keep_original = keep_original
        self.surface_temp_method = surface_temp_method

    def _check_X(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HabitabilityFeatureTransformer expects a pandas DataFrame")
        if self.radius_col not in X.columns:
            raise KeyError(f"required column '{self.radius_col}' not found")
        return X

    def fit(self, X: pd.DataFrame, y: Any = None) -> "HabitabilityFeatureTransformer":
        """Validate ``X`` and record its columns. Returns ``self``."""
        X = self._check_X(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the engineered feature DataFrame (same index as ``X``)."""
        if not hasattr(self, "n_features_in_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("call fit before transform")
        X = self._check_X(X)
        enriched = add_habitability_columns(
            X,
            radius_col=self.radius_col,
            mass_col=self.mass_col,
            density_col=self.density_col,
            flux_col=self.flux_col,
            surface_temp_col=self.surface_temp_col,
            surface_temp_method=self.surface_temp_method,
        )

        def col(name: str) -> pd.Series:
            if name in X.columns:
                return pd.to_numeric(X[name], errors="coerce")
            return pd.Series(np.nan, index=X.index, dtype=float)

        teff, lum, flux = col(self.teff_col), col(self.luminosity_col), col(self.flux_col)
        feats = enriched[
            ["density_eu", "escape_velocity_eu", "esi_2p", "esi_4p_interior", "esi_4p_surface", "esi_4p"]
        ].copy()
        surf = feats["esi_4p_surface"]
        feats["esi_interior_surface_ratio"] = feats["esi_4p_interior"] / surf.where(surf > 0)

        feats["hz_seff_inner"] = kopparapu_seff(teff, "runaway_greenhouse")
        feats["hz_seff_outer"] = kopparapu_seff(teff, "maximum_greenhouse")
        feats["hz_seff_inner_optimistic"] = kopparapu_seff(teff, "recent_venus")
        feats["hz_seff_outer_optimistic"] = kopparapu_seff(teff, "early_mars")
        feats["hz_inner_au"] = hz_distance(lum, feats["hz_seff_inner"], self.luminosity_is_log)
        feats["hz_outer_au"] = hz_distance(lum, feats["hz_seff_outer"], self.luminosity_is_log)

        def within(lo: pd.Series, hi: pd.Series) -> pd.Series:
            known = flux.notna() & lo.notna() & hi.notna()
            inside = (flux >= lo) & (flux <= hi)
            return inside.astype(float).where(known)

        feats["in_conservative_hz"] = within(feats["hz_seff_outer"], feats["hz_seff_inner"])
        feats["in_optimistic_hz"] = within(
            feats["hz_seff_outer_optimistic"], feats["hz_seff_inner_optimistic"]
        )
        feats = feats[FEATURE_NAMES]
        if self.keep_original:
            return pd.concat([X, feats.drop(columns=[c for c in FEATURE_NAMES if c in X.columns])], axis=1)
        return feats

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the output columns."""
        names = list(FEATURE_NAMES)
        if self.keep_original:
            base = list(getattr(self, "feature_names_in_", input_features or []))
            names = base + [n for n in FEATURE_NAMES if n not in base]
        return np.asarray(names, dtype=object)
