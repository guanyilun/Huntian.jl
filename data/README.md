# Input data

| file | what | origin |
|---|---|---|
| `pk_websky.dat` | linear matter P(k) at z = 0, k [h/Mpc], P [(Mpc/h)³], σ₈ = 0.81 | Websky cosmology (Stein et al. 2020) |
| `pk_websky_RAW_unnormalized.dat` | the same P(k) before σ₈ normalization; used by abundance matching for the Tinker mass function | Websky |
| `filters_websky_finecell.dat` | top-hat filter bank (radii in Mpc/h) used by the production configs | Websky filter bank on the 0.852 Mpc/h cell |
| `filters_websky.dat` | coarser filter bank; used by tests | Websky |
| `HomelTab_websky.dat` | homogeneous-ellipsoid collapse table (F, e, p → collapse threshold) | Websky table; `bin/generate_table.jl` regenerates one |
| `websky_halo_mass_completion.txt` | Websky's z-dependent completeness mass M200m_min(z) [Msun/h]; the CIB painter keeps halos above it | Websky public release (`halo_mass_completion.txt`) |
