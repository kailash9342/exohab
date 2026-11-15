"""Self-check for an exohab installation.

Run:  python try_exohab.py

Part 1 (offline) compares exohab with values published in the literature.
Part 2 (needs internet the first time) downloads the NASA Exoplanet Archive,
runs the Monte Carlo and saves three plots as PNG files next to this script.
"""

from pathlib import Path

# Only needed behind a proxy / antivirus HTTPS scanning ("CERTIFICATE_VERIFY_FAILED").
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import exohab as eh

HERE = Path(__file__).resolve().parent
results = []


def check(name, got, expected, tol):
    ok = abs(got - expected) <= tol
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:.4f}, published {expected} (+/- {tol})")


print(f"exohab {eh.__version__}\n")
print("Part 1 - published benchmark values (offline)")
check("Earth ESI (4-parameter)", eh.compute_esi_4p(1, 1, 1, 288).esi, 1.0, 1e-12)
m = eh.compute_esi_4p(radius=0.53, density=0.71, escape_velocity=0.45, surface_temp=240.0)
check("Mars interior ESI  (Kashyap 2017)", m.interior, 0.82, 0.005)
check("Mars surface ESI   (Kashyap 2017)", m.surface, 0.65, 0.005)
check("Mars global ESI    (Kashyap 2017)", m.esi, 0.73, 0.005)
check("Proxima b 2p ESI   (arXiv:1712.01040)", eh.compute_esi_2p(1.12, 0.700522), 0.8692, 5e-4)
check("Earth on Mars scale (MSI, Kashyap 2017)",
      eh.compute_similarity(1, 1, 1, 288, reference="mars").esi, 0.68, 0.005)
check("Sun runaway-greenhouse flux (Kopparapu 2014)", eh.kopparapu_seff(5780.0), 1.107, 1e-9)
check("Earth surface T from T_eq=255 K (Kashyap 2017)", eh.estimate_surface_temp(255.0), 289.28, 0.2)

print("\nPart 2 - live NASA data, Monte Carlo and plots")
try:
    df = eh.load_nasa_exoplanets(surface_temp_method="kashyap2017")
except Exception as exc:  # noqa: BLE001 - report any network problem plainly
    print(f"  [SKIP] could not download data: {exc}")
else:
    print(f"  [PASS] downloaded {len(df)} confirmed planets")
    top = df.nlargest(5, "esi_2p")[["pl_name", "pl_rade", "pl_insol", "esi_2p"]]
    print("  Top 5 by 2-parameter ESI:\n" + top.round(3).to_string(index=False))

    k = df.set_index("pl_name").loc["Kepler-442 b"]
    mc = eh.compute_esi_2p_mc(
        radius=(k.pl_rade, k.pl_radeerr1, k.pl_radeerr2),
        flux=(k.pl_insol, k.pl_insolerr1, k.pl_insolerr2),
        n_samples=20_000, seed=42, truncate=True,
    )
    print(f"  Kepler-442 b ESI: median {mc.median:.3f}, 68% range {mc.ci68[0]:.3f}-{mc.ci68[1]:.3f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [SKIP] plots need matplotlib: pip install "exohab[all]"')
    else:
        for fname, draw in [
            ("check_esi_scatter.png", lambda: eh.plot_esi_scatter(df[df.pl_rade < 2.5])),
            ("check_habitable_zone.png", lambda: eh.plot_habitable_zone(df, color_col="esi_2p")),
            ("check_monte_carlo.png", lambda: eh.plot_mc_distribution(mc, label="ESI (Kepler-442 b)")),
        ]:
            ax = draw()
            ax.figure.savefig(HERE / fname, dpi=110, bbox_inches="tight")
            plt.close(ax.figure)
            print(f"  [PASS] saved {HERE / fname}")

print(f"\n{sum(results)}/{len(results)} benchmark checks passed.")
raise SystemExit(0 if all(results) else 1)
