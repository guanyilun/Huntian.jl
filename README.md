# Huntian.jl

GPU peak-patch simulations of the extragalactic microwave sky. Huntian.jl generates full-sky halo lightcones
with the mass-peak-patch method (Bond & Myers 1996; Stein et al. 2019) and paints correlated maps of CMB
lensing convergence, kinetic and thermal Sunyaev–Zel'dovich effects, the cosmic infrared background and the
integrated Sachs–Wolfe effect. It produces the Huntian (浑天) skies, built to match the Websky simulation
(Stein et al. 2020).

- **Multi-resolution initial conditions:** a global coarse grid plus independently synthesized fine tiles,
  with no global fine-grid FFT. A counter-based RNG (Threefry) makes the field independent of the tiling and
  of the number of GPUs.
- **GPU halo finding** (CUDA.jl): peak finding, radial-shell homogeneous-ellipsoid collapse, Lagrangian
  exclusion and volume reduction, 2LPT displacements and velocities, on the lightcone.
- **Maps:** full-matter line-of-sight field maps (κ, kSZ, τ, ISW) pixelized on the GPU; halo profiles
  (NFW κ, Battaglia tSZ and kSZ) and the CIB (XGPaint) painted from the abundance-matched catalog.
- **Provenance:** every catalog carries its config, seed, code commit and input hashes, so any sky can be
  regenerated exactly.

Units are Mpc/h throughout; see [CONVENTIONS.md](CONVENTIONS.md).

## Installation

Requires Julia 1.11 or later.

```julia
using Pkg
Pkg.add(url="https://github.com/guanyilun/Huntian.jl")
```

CUDA, HDF5 and MPI support are package extensions that load with `using CUDA`, `using HDF5` and
`using MPI, PencilFFTs, PencilArrays`. The production scripts run from the `pipeline/` environment, which
provides Huntian (this repository), CUDA, HDF5 and Healpix:

```sh
julia --project=pipeline -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

A small lightcone octant with the production physics and resolution runs in minutes on one GPU:

```sh
julia --project=pipeline -t 8 bin/huntian.jl examples/config_gpu_octant.toml --verbose
```

This writes `output/catalog_gpu_octant.h5` and the abundance-matched `output/catalog_gpu_octant_AM.h5`.
From Julia:

```julia
using Huntian, TOML, HDF5
cfg = PipelineConfig(TOML.parsefile("config.toml"))
halos = run_multitile_split(cfg; ntile=4, seed=12345, coarse_factor=24)
merged = merge_catalog(halos; volume_reduction=true)

halos, meta, prov = read_halo_catalog("output/catalog_gpu_octant.h5")
cols = read_halo_columns("output/catalog_gpu_octant.h5", (:x, :y, :z, :RTHL))
```

## Halo catalogs

Catalogs are HDF5 files (format `huntian-catalog/1`) with one lossless Float32 dataset per halo field and a
`units` attribute on each. They can be read with any HDF5 library.

| datasets | content | units |
|---|---|---|
| `x`, `y`, `z` | Eulerian comoving position | Mpc/h |
| `vx`, `vy`, `vz` | peculiar velocity (1LPT + 2LPT) | km/s |
| `vx2`, `vy2`, `vz2` | zero in finalized catalogs (the 2LPT displacement before `finalize_eulerian`) | |
| `RTHL` | Lagrangian top-hat radius; M = 4π/3 · 2.775e11 Ω_m · RTHL³ | Mpc/h (M in Msun/h) |
| `overdensity` | mean linear overdensity inside RTHL | |
| `e_v`, `p_v`, `strain_*`, `zform`, `grad*`, `Rf`, … | shear and formation fields (extended records) | |

Root attributes give `n_halos`, `z_out`, `RTHLmax`, `observer` and `mass_definition`. The `/provenance` group
holds the config text, seed, code commit and input hashes. `read_pksc` and `write_pksc` read and write the
Fortran `.pksc` format, for comparison with Websky catalogs.

## Producing a sky

`pipeline/` holds one driver per step; each runs on one node and can be wrapped in any batch system. For a
sky with seed `SEED` written to `SKY`:

```sh
python3 pipeline/make_configs.py SEED SKY                     # SKY/configs/config_oct{000..111}.toml
# per octant ZYX (8 independent jobs each):
julia --project=pipeline -t 32 pipeline/catalog.jl SKY/configs/config_octZYX.toml        # GPUs: all visible
julia --project=pipeline -t 16 pipeline/fieldmap.jl SKY/configs/config_octZYX.toml 4096 SKY/fieldmaps
# once all 8 catalogs exist: one full-sky abundance-matching table, then each octant
julia --project=pipeline pipeline/abundance_match.jl table SKY/catalogs/am_table.txt 10 4.5 SKY/catalogs/catalog_oct*.h5
julia --project=pipeline pipeline/abundance_match.jl apply SKY/catalogs/am_table.txt \
      SKY/catalogs/catalog_octZYX.h5 SKY/catalogs/catalog_octZYX_AM.h5 ZYX
julia --project=pipeline -t 32 pipeline/paint_halos.jl ZYX SKY/catalogs/catalog_octZYX_AM.h5 SKY/halomaps 4096 kappa,tsz,ksz octZYX
julia --project=$XGPAINT -t 32 pipeline/paint_cib.jl ZYX SKY/catalogs/catalog_octZYX_AM.h5 SKY/cibmaps 4096 100,143,217,353,545,857 octZYX
# once all octants are painted
julia --project=pipeline -t 8 pipeline/assemble.jl SKY        # SKY/maps/*.fits
```

| step | resources per job (measured) |
|---|---|
| `catalog.jl` | 2 × A100-80GB, 32 cores; 1.8 h, 94 GB host memory |
| `fieldmap.jl` | 1 × A100-80GB; 1.3 h, 40 GB |
| `abundance_match.jl` | table 19 min, 64 GB; apply 8 min, 58 GB |
| `paint_halos.jl` | 32 cores; 7 min, 29 GB |
| `paint_cib.jl` | 32 cores; 35 min, 27 GB |

`paint_halos.jl` paints the kSZ halo gas with the Battaglia et al. (2016) profile as published (their eq. A1,
`gas=b16`, default). `gas=websky` selects the form used by the Websky map maker (outer exponent −β, and a τ amplitude
whose length unit is the 200ρ̄_m radius of M200c rather than R200c), for comparisons with Websky only. Halo masses for the pressure and gas profiles are M200c
computed from the catalogue's M200m with an NFW profile and the Duffy et al. (2008) concentration (`mass=nfw`,
default). The default tSZ map is uncalibrated; to scale it to an observed amplitude, see the opt-in procedure in
[docs/tsz_calibration.md](docs/tsz_calibration.md) (`massbias=B`).

One sky takes about 39 A100-hours. `paint_cib.jl` runs in the environment of the XGPaint.jl fork at
https://github.com/guanyilun/XGPaint.jl (branch `lensing-kappa`), given as `$XGPAINT`.

## Data

`data/` holds the input power spectrum, filter bank, collapse table and the Websky completeness table; see
[data/README.md](data/README.md).

## Citation

Paper in preparation. The method follows Bond & Myers (1996), Stein, Alvarez & Bond (2019) and Stein et al.
(2020, Websky).
