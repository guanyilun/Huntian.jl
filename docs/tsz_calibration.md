# Calibrating the tSZ amplitude (opt-in)

The default tSZ map is the Battaglia et al. (2012, B12) AGN-feedback pressure profile, as published, painted on
the halo catalogue's true masses (M200c from the abundance-matched M200m via NFW + Duffy et al. 2008) in the
Huntian cosmology (σ₈ = 0.81). It is a prediction, not a fit to data. Its power is about 1.6–1.8× the
foreground-marginalised Planck y spectrum (Bolliet et al. 2018) and 2.1–2.6× the SPT/ACT tSZ amplitudes at ℓ = 3000.
This excess is a property of the model (B12 on true masses with a Planck-like cosmology), not of the painting: the
painted map agrees with a halo model built from the same profile to about 1% at ℓ = 300–4000.

Users who need a tSZ map with an observed amplitude can apply a hydrostatic mass bias B: each halo's pressure profile
(its amplitude P200, its radius R200c and its mass-dependent shape) is evaluated at M200c/B. Halo positions, κ and
kSZ are unchanged. B = 1 is the published model.

## Procedure

1. Choose B (below), or fit your own with a halo model of the same profile.
2. Repaint the tSZ halo maps with the painter option `massbias=B` and assemble them:

       julia --project=pipeline -t 32 pipeline/paint_halos.jl ZYX SKY/catalogs/catalog_octZYX_AM.h5 SKY/halomaps_B 4096 tsz octZYX massbias=1.16
       # for all 8 octants, then sum the 8 tsz_y maps (pipeline/assemble.jl does this for the default maps)

   tSZ-only painting takes a few minutes per octant on 32 cores.
3. Record B with the map; maps painted with B ≠ 1 are calibrated products, not predictions.

## Reference values

Fitted with the halo model to the Bolliet et al. (2018) foreground-marginalised Planck y bandpowers (their Table 4),
diagonal errors:

| fit | B |
|---|---|
| ℓ_eff ≥ 250 (preferred: lower multipoles are set by a few nearby clusters and single-sky variance) | **1.16 ± 0.08** |
| all bins (ℓ_eff 10–960) | 1.15 |
| ℓ_eff ≥ 500 | 1.21 |

The ±0.08 propagates Bolliet et al.'s full-likelihood uncertainty; the diagonal-error interval (±0.04) is too narrow
because the bandpower errors are correlated. With B = 1.16 the map remains about 1.5–1.7× the SPT/ACT tSZ
amplitudes at ℓ = 3000; matching those would need B ≈ 1.3–1.4, which the Planck spectrum disfavours (χ² 0.8 → 13.4
for 5 degrees of freedom). No single B matches both in this model.

Power relative to B = 1 (halo model; multiply a B = 1 spectrum by these factors):

| B | ℓ = 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|
| 1.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.1 | 0.745 | 0.750 | 0.756 | 0.765 |
| 1.16 | 0.633 | 0.639 | 0.646 | 0.659 |
| 1.2 | 0.570 | 0.577 | 0.585 | 0.600 |
| 1.3 | 0.445 | 0.453 | 0.462 | 0.479 |
| 1.4 | 0.354 | 0.363 | 0.372 | 0.389 |
| 1.5 | 0.286 | 0.294 | 0.303 | 0.320 |
| 2.0 | 0.118 | 0.124 | 0.130 | 0.142 |

The response is close to C_ℓ ∝ B^−3. These factors are for the NFW mass conversion (the default, used by the
Huntian v0.1 maps).
