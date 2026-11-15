"""One-line plots for habitability results.

Requires the optional dependency ``matplotlib`` (``pip install exohab[viz]`` or
``exohab[all]``). Importing this module never fails; calling a plot function
without matplotlib raises a clear ImportError.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .habitable_zone import TEFF_RANGE, kopparapu_seff
from .uncertainty import MCResult

__all__ = ["plot_esi_scatter", "plot_mc_distribution", "plot_habitable_zone"]


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "exohab plotting requires matplotlib. Install it with: pip install exohab[all]"
        ) from exc
    return plt


def _label_points(
    ax: Any, x: np.ndarray, y: np.ndarray, labels: list, min_sep: float = 0.045
) -> None:
    """Label points in a stacked column with leader lines so labels never overlap.

    Labels are placed (in data coordinates of a [0, 1] axis) in a column to the
    left of the left-most labelled point, ordered by y and spaced at least
    ``min_sep`` apart.
    """
    if len(labels) == 0:
        return
    order = np.argsort(-y)
    x_text = max(0.02, float(np.min(x)) - 0.33)
    placed: list = []
    for i in order:
        target = min(float(y[i]), 1.0) if not placed else min(float(y[i]), placed[-1] - min_sep)
        placed.append(target)
    # shift the whole column up if it ran below the axis
    shift = max(0.0, 0.03 - placed[-1])
    for i, yt in zip(order, placed):
        ax.annotate(
            labels[i],
            xy=(x[i], y[i]),
            xytext=(x_text, yt + shift),
            textcoords="data",
            fontsize=7,
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
            arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.6},
        )


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
        _label_points(
            ax,
            top[interior_col].to_numpy(float),
            top[surface_col].to_numpy(float),
            top[name_col].astype(str).tolist(),
        )

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


def plot_habitable_zone(
    df: pd.DataFrame,
    flux_col: str = "pl_insol",
    teff_col: str = "st_teff",
    name_col: Optional[str] = "pl_name",
    labels: Optional[list] = None,
    color_col: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """Stellar effective temperature vs. insolation with Kopparapu (2014) HZ bands.

    The conservative habitable zone (runaway to maximum greenhouse) is shaded
    dark green and the optimistic one (recent Venus to early Mars) light green,
    over the 2600-7200 K range where the fits are valid. The insolation axis is
    logarithmic and inverted (hot planets on the left), as in Kopparapu et al.
    (2013, 2014).

    Parameters
    ----------
    df : DataFrame
        Table with insolation (Earth units) and stellar effective temperature.
    flux_col, teff_col : str
        Column names.
    name_col : str, optional
        Column holding planet names, used with ``labels``.
    labels : list of str, optional
        Planet names to annotate.
    color_col : str, optional
        Column used to colour the points (e.g. ``"esi_2p"``).
    ax : matplotlib Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _pyplot()
    for c in (flux_col, teff_col):
        if c not in df.columns:
            raise KeyError(f"column '{c}' not found")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))

    teff = np.linspace(TEFF_RANGE[0], TEFF_RANGE[1], 200)
    ax.fill_betweenx(teff, kopparapu_seff(teff, "recent_venus"), kopparapu_seff(teff, "early_mars"),
                     color="tab:green", alpha=0.15, lw=0, label="Optimistic HZ")
    ax.fill_betweenx(teff, kopparapu_seff(teff, "runaway_greenhouse"),
                     kopparapu_seff(teff, "maximum_greenhouse"),
                     color="tab:green", alpha=0.35, lw=0, label="Conservative HZ")

    data = df.copy()
    data[flux_col] = pd.to_numeric(data[flux_col], errors="coerce")
    data[teff_col] = pd.to_numeric(data[teff_col], errors="coerce")
    data = data[(data[flux_col] > 0) & data[teff_col].notna()]
    kwargs: dict = {"s": 12, "alpha": 0.8, "edgecolors": "none"}
    if color_col and color_col in data.columns:
        sc = ax.scatter(data[flux_col], data[teff_col], c=data[color_col], cmap="viridis",
                        vmin=0, vmax=1, **kwargs)
        ax.figure.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(data[flux_col], data[teff_col], color="0.3", **kwargs)
    ax.scatter([1.0], [5772.0], marker="*", s=150, color="tab:blue", zorder=3, label="Earth")

    if labels and name_col and name_col in data.columns:
        for _, row in data[data[name_col].isin(labels)].iterrows():
            ax.annotate(str(row[name_col]), (row[flux_col], row[teff_col]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)

    ax.set_xscale("log")
    lo = float(np.nanmin(data[flux_col])) if len(data) else 0.1
    hi = float(np.nanmax(data[flux_col])) if len(data) else 10.0
    ax.set_xlim(max(hi, 3.0) * 1.2, min(lo, 0.2) / 1.2)  # inverted: hot on the left
    ax.set_ylim(2400, 7400)
    ax.set_xlabel("Insolation (Earth = 1)")
    ax.set_ylabel("Stellar effective temperature (K)")
    ax.set_title("Habitable zone (Kopparapu et al. 2014)")
    ax.legend(loc="upper right", fontsize=8)
    return ax
