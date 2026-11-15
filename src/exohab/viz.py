"""One-line plots for habitability results.

Requires the optional dependency ``matplotlib`` (``pip install exohab[viz]`` or
``exohab[all]``). Importing this module never fails; calling a plot function
without matplotlib raises a clear ImportError.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .uncertainty import MCResult

__all__ = ["plot_esi_scatter", "plot_mc_distribution"]


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "exohab plotting requires matplotlib. Install it with: pip install exohab[all]"
        ) from exc
    return plt


def plot_esi_scatter(
    df: pd.DataFrame,
    interior_col: str = "esi_4p_interior",
    surface_col: str = "esi_4p_surface",
    esi_col: Optional[str] = "esi_4p",
    name_col: Optional[str] = "pl_name",
    top_n: int = 10,
    threshold: float = 0.8,
    isolines: tuple = (0.6, 0.8),
    ax: Optional[Any] = None,
) -> Any:
    """Interior vs. surface ESI scatter with quadrants and labelled top candidates.

    Dashed lines at ``threshold`` split the plane into quadrants (Earth-like
    interior and/or surface). Dotted curves are isolines of constant global
    ESI, ``surface = ESI**2 / interior`` since ``ESI = sqrt(I * S)``.

    Parameters
    ----------
    df : DataFrame
        Table with interior and surface ESI columns.
    interior_col, surface_col : str
        Column names of the sub-indices.
    esi_col : str, optional
        Global ESI column used for colouring and ranking; computed as
        ``sqrt(I * S)`` if missing or None.
    name_col : str, optional
        Column with labels for the top candidates.
    top_n : int, default 10
        Number of highest-ESI rows to label (0 disables labels).
    threshold : float, default 0.8
        Quadrant divider (PHL "Earth-like" threshold).
    isolines : tuple of float
        Global-ESI levels drawn as isolines.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _pyplot()
    for c in (interior_col, surface_col):
        if c not in df.columns:
            raise KeyError(f"column '{c}' not found")
    data = df.copy()
    if esi_col is None or esi_col not in data.columns:
        esi_col = "_esi_global"
        data[esi_col] = np.sqrt(data[interior_col] * data[surface_col])
    data = data.dropna(subset=[interior_col, surface_col])

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        data[interior_col],
        data[surface_col],
        c=data[esi_col],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=14,
        alpha=0.8,
        edgecolors="none",
    )
    ax.figure.colorbar(sc, ax=ax, label="Global ESI")
    ax.axvline(threshold, color="0.4", ls="--", lw=1)
    ax.axhline(threshold, color="0.4", ls="--", lw=1)
    xi = np.linspace(1e-3, 1, 400)
    for level in isolines:
        ys = level**2 / xi
        m = ys <= 1
        ax.plot(xi[m], ys[m], color="0.6", ls=":", lw=1)
        ax.annotate(f"ESI={level:g}", (xi[m][0], 1.0), fontsize=7, color="0.4",
                    ha="left", va="bottom")
    ax.scatter([1], [1], marker="*", s=150, color="tab:blue", zorder=3, label="Earth")

    if top_n and name_col and name_col in data.columns:
        top = data.nlargest(int(top_n), esi_col)
        for _, row in top.iterrows():
            ax.annotate(str(row[name_col]), (row[interior_col], row[surface_col]),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Interior ESI (radius, density)")
    ax.set_ylabel("Surface ESI (escape velocity, temperature)")
    ax.set_title("Interior vs. surface Earth Similarity Index")
    ax.legend(loc="lower left", fontsize=8)
    return ax


def plot_mc_distribution(
    result: MCResult,
    bins: int = 60,
    label: str = "ESI",
    ax: Optional[Any] = None,
) -> Any:
    """Histogram of Monte Carlo samples with median and 68% / 95% intervals.

    Parameters
    ----------
    result : MCResult
        Output of :func:`exohab.compute_esi_mc` or :func:`exohab.compute_esi_2p_mc`.
    bins : int, default 60
        Number of histogram bins.
    label : str, default "ESI"
        x-axis label.
    ax : matplotlib Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _pyplot()
    if not isinstance(result, MCResult):
        raise TypeError("result must be an MCResult")
    if result.n_valid == 0:
        raise ValueError("result contains no valid samples")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(result.samples, bins=bins, density=True, color="tab:blue", alpha=0.6)
    ax.axvspan(*result.ci95, color="tab:orange", alpha=0.12, label="95% interval")
    ax.axvspan(*result.ci68, color="tab:orange", alpha=0.25, label="68% interval")
    ax.axvline(result.median, color="k", lw=1.5, label=f"median = {result.median:.3f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Probability density")
    ax.set_title(f"Monte Carlo distribution (n = {result.n_valid})")
    ax.legend(fontsize=8)
    return ax
