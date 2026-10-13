# Units and coordinates

**Huntian works in Mpc/h for all lengths.** The Fortran peak-patch code (`hpkvd`) and the public Websky
catalogs use **Mpc**. The two differ by a factor of h.

To match a box specified as `L` Mpc, set `boxsize = L * h` (Mpc/h). Websky with h = 0.68:
7700 Mpc × 0.68 = 5236 Mpc/h, and 5236 / 6144 = 0.852 Mpc/h cells (= 1.2533 Mpc).

| quantity | Huntian | Fortran / Websky |
|---|---|---|
| `boxsize`, cell size, observer `cenx/y/z` | **Mpc/h** | Mpc |
| halo positions `x, y, z` | **Mpc/h** | Mpc |
| `RTHL` (Lagrangian top-hat radius) | **Mpc/h** | Mpc |
| filter radii in the filter bank | **Mpc/h** | Mpc |
| `k`, `P(k)` | h/Mpc, (Mpc/h)³ | same |
| comoving distance `chi(z)` | **Mpc/h** (H0 = 100h) | Mpc |
| velocities (finalized catalogs) | km/s | km/s |
| halo mass `M = 4π/3 · 2.775e11 Ω_m · RTHL³` | **Msun/h** | Msun with R in Mpc and ρ_m = 2.775e11 Ω_m h² |

Mpc/h is required, not a choice: `Cosmology.chi(z)` returns (c/100)∫dz/E in Mpc/h, and the lightcone assigns
each halo its redshift from its distance through `chi`, so positions and box must be in Mpc/h.

**Comparing with a Websky catalog:** multiply Websky lengths by h first (`r[Mpc/h] = r[Mpc] · h`,
`R[Mpc/h] = R[Mpc] · h`); then the same mass formula gives Msun/h, i.e. M[Msun/h] = M[Msun] · h.

**Check at run time:** with `verbose=true`, `run_multitile*` prints the cell size and box in both units,
e.g. `cellsize=0.8522 Mpc/h (=1.2533 Mpc), full box=5236.0 Mpc/h (=7700.0 Mpc)`.
