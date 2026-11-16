---
description: Exact formulas, reference values, validation results and limitations behind exohab - Earth Similarity Index, Kopparapu habitable zone, PHI, BCI and Monte Carlo uncertainty.
---

# Science & methods

This page lists exactly what exohab computes, where each formula comes from,
how the results were checked and what they do **not** mean.

## Earth Similarity Index (four parameters)

Schulze-Makuch et al. (2011) define, for each planetary property \(x\) with
Earth reference value \(x_0\) and weight exponent \(w\):

\[
\mathrm{ESI}_x = \left(1 - \left|\frac{x - x_0}{x + x_0}\right|\right)^{w}
\]

The four properties are combined as geometric means:

\[
\mathrm{ESI}_I = \sqrt{\mathrm{ESI}_R\,\mathrm{ESI}_\rho}, \qquad
\mathrm{ESI}_S = \sqrt{\mathrm{ESI}_{v_e}\,\mathrm{ESI}_{T_s}}, \qquad
\mathrm{ESI} = \sqrt{\mathrm{ESI}_I\,\mathrm{ESI}_S}
\]

This is algebraically identical to the Planetary Habitability Laboratory form
\(\prod_i (1 - |\ldots|)^{w_i/n}\) with \(n = 4\).

| Property | Earth reference | Weight \(w\) |
|---|---|---|
| Mean radius \(R\) | 1 EU (6371 km) | 0.57 |
| Bulk density \(\rho\) | 1 EU (5.51 g cm⁻³) | 1.07 |
| Escape velocity \(v_e\) | 1 EU (11.19 km s⁻¹) | 0.70 |
| Mean surface temperature \(T_s\) | 288 K | 5.58 |

## Earth Similarity Index (two parameters)

For planets with only a radius \(R\) and a stellar flux \(S\) (Earth units),
the Planetary Habitability Laboratory uses:

\[
\mathrm{ESI} = 1 - \sqrt{\tfrac{1}{2}\left[\left(\frac{S - 1}{S + 1}\right)^2 + \left(\frac{R - 1}{R + 1}\right)^2\right]}
\]

This is not numerically identical to the four-parameter ESI.

## Similarity to other bodies

`compute_similarity` applies the four-parameter formula with another body's
reference values. The Mars preset is the **Mars Similarity Index** of Kashyap
Jagadeesh et al. (2017), with its own published reference values (3390 km,
3.93 g cm⁻³, 5.03 km s⁻¹, 240 K) and weights (0.86, 2.10, 1.09, 3.23). Venus
and Titan use NASA fact-sheet values with the Earth ESI weights, because no
body-specific weights have been published.

## Surface temperature

Catalogues list equilibrium temperature \(T_\mathrm{eq}\), not surface
temperature. Kashyap Jagadeesh (2017, eq. 3.12) fitted the following relation
to Earth, Mars, the Moon, Titan and Triton (Venus was excluded):

\[
T_s = 9.650 + 1.096\,T_\mathrm{eq}
\]

It only applies to rocky planets with an atmosphere, rests on five bodies, and
is an extrapolation above about 255 K. exohab uses it only when you ask for
`surface_temp_method="kashyap2017"`.

## Habitable zone

Kopparapu et al. (2014) give each boundary as a polynomial in
\(T_\star = T_\mathrm{eff} - 5780\,\mathrm{K}\):

\[
S_\mathrm{eff} = S_{\mathrm{eff},\odot} + a T_\star + b T_\star^2 + c T_\star^3 + d T_\star^4,
\qquad d = \sqrt{L / S_\mathrm{eff}}\ \mathrm{AU}
\]

The coefficients are copied exactly from the authors' published calculator
for 1 Earth-mass planets, plus the 5 and 0.1 Earth-mass runaway-greenhouse
limits. The fits are valid for 2600–7200 K.

## PHI and BCI

\[
\mathrm{PHI} = (S \cdot E \cdot C \cdot L)^{1/4} \quad \text{(Schulze-Makuch et al. 2011)}, \qquad
\mathrm{BCI} = (S \cdot E \cdot T \cdot G \cdot A)^{1/5} \quad \text{(Irwin et al. 2014)}
\]

The factor scores in [0, 1] come from expert rubrics in those papers. exohab
implements only how the scores are combined.

## Uncertainty

A quote \(x^{+a}_{-b}\) is sampled from the two-piece normal distribution: a
normal with width \(b\) below the quoted value and \(a\) above it. A draw falls
below the quoted value with probability \(b/(a+b)\), and the mean is
\(x + \sqrt{2/\pi}\,(a - b)\) (Wallis 2014). Inputs are sampled
independently, except that density and escape velocity are derived from the
same mass and radius in each draw when `mass` is given.

## Validation

The automated test suite checks exohab against these published values, among
others:

| Check | Published | exohab |
|---|---|---|
| Mars, four-parameter ESI (interior / surface / global) | 0.82 / 0.65 / 0.73 (Kashyap Jagadeesh 2017) | 0.815 / 0.649 / 0.727 |
| Proxima b, two-parameter ESI | 0.8692 (arXiv:1712.01040) | 0.8692 |
| Earth on the Mars Similarity Index | 0.68 (Kashyap Jagadeesh 2017) | 0.683 |
| Kepler-442 b, four-parameter ESI | 0.94 / 0.84 / 0.89 (Kashyap Jagadeesh 2017) | within rounding |
| Habitable-zone flux at 5780 K | 1.107 and 0.356 (Kopparapu et al. 2014) | exact |
| Earth's surface temperature from \(T_\mathrm{eq}\) = 255 K | 289.28 K (Kashyap Jagadeesh 2017) | 289.13 K (rounded coefficients) |

## Limitations

- **ESI measures similarity, not the probability of life.** A high ESI means
  Earth-like bulk properties and insolation, nothing more.
- **Equilibrium vs. surface temperature.** By default `pl_eqt` is used
  directly as the temperature input, which lowers the surface ESI of temperate
  planets.
- **Mass.** The archive's `pl_bmasse` may be a minimum mass (M sin i) for
  radial-velocity planets.
- **Missing error bars** are treated as zero and flagged; those intervals are
  lower limits.
- **Habitable-zone limits** are NaN outside 2600–7200 K unless extrapolation
  is requested explicitly.

## References

- Schulze-Makuch, D. et al. (2011). A Two-Tiered Approach to Assessing the Habitability of Exoplanets. *Astrobiology* 11(10), 1041–1052. doi:10.1089/ast.2010.0592
- Planetary Habitability Laboratory, UPR Arecibo. Earth Similarity Index. <https://phl.upr.edu/projects/earth-similarity-index-esi>
- Kashyap Jagadeesh, M. (2017). *Earth Similarity Index and Habitability Studies of Exoplanets.* M.Phil. dissertation, Christ University. arXiv:1801.07101
- Irwin, L. N., Méndez, A., Fairén, A. G., Schulze-Makuch, D. (2014). Assessing the Possibility of Biological Complexity on Other Worlds. *Challenges* 5(1), 159–174. doi:10.3390/challe5010159
- Kopparapu, R. K. et al. (2013). Habitable Zones around Main-sequence Stars: New Estimates. *ApJ* 765, 131. arXiv:1301.6674
- Kopparapu, R. K. et al. (2014). Habitable Zones around Main-sequence Stars: Dependence on Planetary Mass. *ApJL* 787, L29. arXiv:1404.5292
- Wallis, K. F. (2014). The Two-Piece Normal, Binormal, or Double Gaussian Distribution. *Statistical Science* 29(1), 106–112.
