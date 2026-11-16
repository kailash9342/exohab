"""A guided tour that runs every public function of exohab.

Run:  python full_tour.py

It prints what each function does and checks the result, saves plots in an
``exohab_tour_output`` folder in the current directory, and finishes by
confirming that every name in ``exohab.__all__`` was exercised.

Sections 1-5 work offline. Section 6 onwards downloads the NASA Exoplanet
Archive the first time (about 5 MB), then uses the local cache.
Plots need matplotlib and the ML section needs scikit-learn
(``pip install "exohab[all]"``); those sections are skipped if missing.
"""

from __future__ import annotations

import math
from pathlib import Path

# Only needed behind a proxy / antivirus HTTPS scanning ("CERTIFICATE_VERIFY_FAILED").
# It uses the operating system's certificate store; it does not disable checks.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import pandas as pd

import exohab as eh

OUT = Path.cwd() / "exohab_tour_output"
used: set = set()
passed = failed = skipped = 0


def use(*names: str) -> None:
    """Record which public names were exercised."""
    used.update(names)


def check(label: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    passed += bool(ok)
    failed += not ok
    print(f"   [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ->  {detail}" if detail else ""))


def skip(label: str, why: str) -> None:
    global skipped
    skipped += 1
    print(f"   [SKIP] {label}: {why}")


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


use("__version__")
print(f"exohab {eh.__version__} - full tour of every public function")

# ---------------------------------------------------------------------------
section("1. Earth Similarity Index (ESI)")
# ---------------------------------------------------------------------------
use("ESI_WEIGHTS", "compute_esi_4p", "ESIResult", "compute_esi_2p")
print(f"   ESI weights (Schulze-Makuch et al. 2011): {eh.ESI_WEIGHTS}")

earth = eh.compute_esi_4p(radius=1.0, density=1.0, escape_velocity=1.0, surface_temp=288.0)
check("Earth compared with itself scores exactly 1", earth.esi == 1.0, f"{earth}")
check("compute_esi_4p returns an ESIResult", isinstance(earth, eh.ESIResult))

mars = eh.compute_esi_4p(radius=0.53, density=0.71, escape_velocity=0.45, surface_temp=240.0)
check("Mars matches Kashyap (2017): interior 0.82, surface 0.65, ESI 0.73",
      abs(mars.interior - 0.82) < 0.005 and abs(mars.surface - 0.65) < 0.005
      and abs(mars.esi - 0.73) < 0.005,
      f"{mars.interior:.3f} / {mars.surface:.3f} / {mars.esi:.3f}")

proxima = eh.compute_esi_2p(radius=1.12, flux=0.700522)
check("2-parameter ESI of Proxima b matches arXiv:1712.01040 (0.8692)",
      abs(proxima - 0.8692) < 5e-4, f"{proxima:.4f}")

arr = eh.compute_esi_2p(radius=[1.0, 0.532, 11.2], flux=[1.0, 0.431, 0.037])
check("works on lists/arrays (Earth, Mars, Jupiter)", arr.shape == (3,), np.round(arr, 3).tolist())
ser = eh.compute_esi_2p(pd.Series([1.0, -1.0], index=["ok", "bad"]), 1.0)
check("pandas Series keep their index; invalid input gives NaN",
      list(ser.index) == ["ok", "bad"] and math.isnan(ser["bad"]), ser.round(3).to_dict())

# ---------------------------------------------------------------------------
section("2. Similarity to other worlds")
# ---------------------------------------------------------------------------
use("REFERENCE_BODIES", "compute_similarity")
print(f"   Built-in reference bodies: {sorted(eh.REFERENCE_BODIES)}")
for body in sorted(eh.REFERENCE_BODIES):
    print(f"     - {body}: {eh.REFERENCE_BODIES[body]['source']}")

msi = eh.compute_similarity(1.0, 1.0, 1.0, 288.0, reference="mars")
check("Earth on the Mars Similarity Index scale is 0.68 (Kashyap 2017)",
      abs(msi.esi - 0.68) < 0.005, f"{msi.esi:.3f}")
v = eh.REFERENCE_BODIES["venus"]
venus = eh.compute_esi_4p(v["radius"], v["density"], v["escape_velocity"], v["surface_temp"])
check("Venus: Earth's twin inside (interior > 0.95), hostile at the surface (< 0.3)",
      venus.interior > 0.95 and venus.surface < 0.3,
      f"interior {venus.interior:.2f}, surface {venus.surface:.2f}")
custom = {"radius": 1.5, "density": 1.0, "escape_velocity": 1.2, "surface_temp": 300.0}
same = eh.compute_similarity(1.5, 1.0, 1.2, 300.0, reference=custom)
check("a custom reference body scores 1 against itself", abs(same.esi - 1) < 1e-12)

# ---------------------------------------------------------------------------
section("3. Derived quantities, surface temperature, PHI and BCI")
# ---------------------------------------------------------------------------
use("density_from_mass_radius", "escape_velocity_from_mass_radius",
    "estimate_surface_temp", "SURFACE_TEMP_METHODS", "compute_phi", "compute_bci")
rho = eh.density_from_mass_radius(mass=0.107, radius=0.532)
ve = eh.escape_velocity_from_mass_radius(mass=0.107, radius=0.532)
check("Mars density and escape velocity from mass and radius (~0.71, ~0.45 Earth units)",
      abs(rho - 0.71) < 0.01 and abs(ve - 0.45) < 0.01, f"{rho:.3f}, {ve:.3f}")

print(f"   Surface-temperature methods: {list(eh.SURFACE_TEMP_METHODS)}")
ts = eh.estimate_surface_temp(255.0)
check("Earth: T_eq 255 K -> estimated surface 289 K (Kashyap 2017 table: 289.28 K)",
      abs(ts - 289.28) < 0.2, f"{ts:.2f} K")

print("   PHI and BCI combine factor scores (0-1) that YOU assign from the papers' rubrics.")
print("   The scores below are illustrative only, not published values.")
phi = eh.compute_phi(substrate=0.8, energy=0.9, chemistry=0.7, liquid=0.6)
check("PHI is the geometric mean of 4 scores", abs(phi - (0.8 * 0.9 * 0.7 * 0.6) ** 0.25) < 1e-12,
      f"{phi:.3f}")
bci = eh.compute_bci(substrate=[0.9, 0.5], energy=[0.8, 0.5], temperature=[0.7, 0.4],
                     geophysics=[0.9, 0.3], age=[1.0, 0.6], relative=True)
check("BCI_rel: the best world in the set scores 1", abs(bci[0] - 1) < 1e-12, np.round(bci, 3).tolist())

# ---------------------------------------------------------------------------
section("4. Habitable zone (Kopparapu et al. 2014)")
# ---------------------------------------------------------------------------
use("KOPPARAPU_2014", "kopparapu_seff", "hz_distance")
print(f"   Limits available: {list(eh.KOPPARAPU_2014)}")
inner = eh.kopparapu_seff(5780.0, "runaway_greenhouse")
outer = eh.kopparapu_seff(5780.0, "maximum_greenhouse")
d_in, d_out = eh.hz_distance(1.0, inner), eh.hz_distance(1.0, outer)
check("Sun-like star: conservative HZ from 0.95 to 1.68 AU, and Earth (1 AU) is inside",
      d_in < 1.0 < d_out, f"{d_in:.3f} - {d_out:.3f} AU")
check("outside the valid 2600-7200 K range the result is NaN (no silent extrapolation)",
      math.isnan(eh.kopparapu_seff(2500.0)))

# ---------------------------------------------------------------------------
section("5. Uncertainty: Monte Carlo with asymmetric error bars")
# ---------------------------------------------------------------------------
use("sample_split_normal", "split_normal_cdf", "compute_esi_2p_mc", "compute_esi_mc", "MCResult")
draws = eh.sample_split_normal(1.0, 0.3, 0.1, 100_000, np.random.default_rng(0))
frac_below = float(np.mean(draws < 1.0))
check("split normal 1.0 +0.3/-0.1: 25% of draws fall below the quoted value",
      abs(frac_below - 0.25) < 0.01 and abs(eh.split_normal_cdf(1.0, 1.0, 0.3, 0.1) - 0.25) < 1e-12,
      f"{frac_below:.3f}")

mc2 = eh.compute_esi_2p_mc(radius=(1.34, 0.11, -0.18), flux=(0.66, 0.23, -0.41),
                           n_samples=20_000, seed=42, truncate=True)
check("Kepler-442 b ESI distribution (MCResult, all 20,000 samples kept)",
      isinstance(mc2, eh.MCResult) and mc2.n_valid == 20_000,
      f"median {mc2.median:.3f}, 68% {mc2.ci68[0]:.3f}-{mc2.ci68[1]:.3f}")

mc4 = eh.compute_esi_mc(radius=(0.920, 0.013, -0.012), mass=(0.692, 0.022, -0.022),
                        surface_temp=(eh.estimate_surface_temp(249.7), 1.096 * 2.4),
                        n_samples=20_000, seed=42)
check("TRAPPIST-1 e 4-parameter ESI with interior/surface components",
      mc4.ci95[0] <= mc4.median <= mc4.ci95[1] and set(mc4.components) == {"interior", "surface"},
      f"median {mc4.median:.3f}, 95% {mc4.ci95[0]:.3f}-{mc4.ci95[1]:.3f}")

# ---------------------------------------------------------------------------
section("6. Live NASA Exoplanet Archive data")
# ---------------------------------------------------------------------------
use("load_nasa_exoplanets", "add_habitability_columns", "compute_esi_mc_catalog")
df = None
try:
    df = eh.load_nasa_exoplanets(surface_temp_method="kashyap2017")
except Exception as exc:  # noqa: BLE001 - report network problems plainly
    skip("download", str(exc)[:200])

if df is not None:
    check("downloaded the confirmed-planet catalogue", len(df) > 5000, f"{len(df)} planets")
    print("   Top 5 by 2-parameter ESI:")
    print(df.nlargest(5, "esi_2p")[["pl_name", "pl_rade", "pl_insol", "esi_2p", "esi_4p"]]
          .round(3).to_string(index=False))
    raw = df.drop(columns=[c for c in df.columns if c.startswith(("esi_", "density_eu",
                                                                  "escape_velocity_eu",
                                                                  "surface_temp_est"))])
    again = eh.add_habitability_columns(raw, surface_temp_method="kashyap2017")
    check("add_habitability_columns reproduces the loader's ESI columns",
          np.allclose(again["esi_2p"], df["esi_2p"], equal_nan=True))
    unc = eh.compute_esi_mc_catalog(df, method="2p", n_samples=1000, seed=0)
    n_ok = int((unc["esi_2p_mc_n_valid"] > 0).sum())
    check("Monte Carlo ESI for the whole catalogue", n_ok > 5000, f"{n_ok} planets with a result")
else:
    use("add_habitability_columns", "compute_esi_mc_catalog")
    skip("catalogue sections", "no data")

# ---------------------------------------------------------------------------
section("7. Machine-learning features (scikit-learn)")
# ---------------------------------------------------------------------------
use("HabitabilityFeatureTransformer")
try:
    import sklearn  # noqa: F401
except ImportError:
    skip("ML features", 'needs scikit-learn: pip install "exohab[all]"')
else:
    if df is not None:
        X = eh.HabitabilityFeatureTransformer(surface_temp_method="kashyap2017").fit_transform(df)
        check("15 engineered features per planet", X.shape[1] == 15,
              f"{X.shape}; {int(X['in_conservative_hz'].sum())} planets in the conservative HZ")
    else:
        skip("ML features", "no data")

# ---------------------------------------------------------------------------
section("8. Plots (matplotlib)")
# ---------------------------------------------------------------------------
use("plot_esi_scatter", "plot_habitable_zone", "plot_mc_distribution")
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    skip("plots", 'need matplotlib: pip install "exohab[all]"')
else:
    OUT.mkdir(exist_ok=True)
    plots = [("monte_carlo.png", lambda: eh.plot_mc_distribution(mc4, label="ESI (TRAPPIST-1 e)"))]
    if df is not None:
        plots += [
            ("esi_scatter.png", lambda: eh.plot_esi_scatter(df[df.pl_rade < 2.5])),
            ("habitable_zone.png", lambda: eh.plot_habitable_zone(
                df, color_col="esi_2p", labels=["Kepler-442 b", "TOI-700 d"])),
        ]
    for fname, draw in plots:
        ax = draw()
        ax.figure.savefig(OUT / fname, dpi=110, bbox_inches="tight")
        plt.close(ax.figure)
        check(f"saved {fname}", (OUT / fname).stat().st_size > 10_000, str(OUT / fname))

# ---------------------------------------------------------------------------
section("Summary")
# ---------------------------------------------------------------------------
missing = sorted(set(eh.__all__) - used)
check(f"every public name exercised ({len(set(eh.__all__) & used)}/{len(eh.__all__)})",
      not missing, f"missing: {missing}" if missing else "")
print(f"\n   {passed} passed, {failed} failed, {skipped} skipped")
raise SystemExit(1 if failed else 0)
