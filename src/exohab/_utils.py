"""Internal helpers for vectorised input handling (not part of the public API)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[float, int, Sequence[float], np.ndarray, pd.Series]


def to_array(x: Any) -> np.ndarray:
    """Convert scalars, lists, arrays or Series to a float ndarray.

    ``None`` becomes ``nan`` so missing catalogue values propagate as NaN.
    """
    if x is None:
        return np.asarray(np.nan, dtype=float)
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def series_index(*args: Any) -> Optional[pd.Index]:
    """Return the index of the first pandas Series among ``args`` (or None)."""
    for a in args:
        if isinstance(a, pd.Series):
            return a.index
    return None


def restore(arr: np.ndarray, index: Optional[pd.Index], name: Optional[str] = None) -> Any:
    """Return ``arr`` as a Series (if an input was a Series), a float (0-d) or ndarray."""
    arr = np.asarray(arr, dtype=float)
    if index is not None:
        if arr.ndim == 0:
            arr = np.full(len(index), float(arr))
        return pd.Series(arr, index=index, name=name)
    if arr.ndim == 0:
        return float(arr)
    return arr
