#!/usr/bin/env python3
"""Write the 8 octant configs of one Huntian sky.

    python3 pipeline/make_configs.py SEED SKYDIR

Geometry (Websky's): a periodic 5236 Mpc/h = 7700 Mpc box on a 6144^3 grid (0.852 Mpc/h cells), 16^3 tiles
with periodic cores (N = nsub * ntile), nbuff = 25. The observer sits at a box corner; each octant config
places it at (+-2618, +-2618, +-2618) Mpc/h in tile-centred coordinates (bit 1 -> +), so the 8 octants are
8 views of one periodic box and together cover the full sky to z = 4.5.

Writes SKYDIR/configs/config_octZYX.toml with catalogs going to SKYDIR/catalogs/. Input file paths are
relative to the repository root (jobs run from there).
"""
import os
import sys

N, NTILE, NBUFF = 6144, 16, 25
NSUB = N // NTILE
NMESH = NSUB + 2 * NBUFF
CELL = 5236.0 / N
OBS = N * CELL / 2

TEMPLATE = """# Huntian sky, seed {seed}, octant {oct}: observer at ({cenx}, {ceny}, {cenz}) Mpc/h (tile-centred
# coordinates) = a box corner. Written by pipeline/make_configs.py; do not hand-edit.

[cosmology]
Om   = 0.31
OB   = 0.049
OL   = 0.69
h    = 0.68

[grid]
n       = {nmesh}
boxsize = {tile:.5f}
nbuff   = {nbuff}
periodic_cores = true
cenx    = {cenx}
ceny    = {ceny}
cenz    = {cenz}

[run]
gaussian_split = true
z_out     = 0.0
z_max     = 4.5
ievol     = 1
ilpt      = 2
ioutshear = 1
wsmooth   = 1
rmax2rs   = 0.0
NonGauss  = 0
fNL       = 0.0
seed      = {seed}
ntile     = {ntile}
coarse_factor = 32
coarse_compensation = true
peak_threshold_per_tile = true
volume_reduction = true
generate_table = false

[files]
pk         = "data/pk_websky.dat"
filterbank = "data/filters_websky_finecell.dat"
homeltab   = "data/HomelTab_websky.dat"
output     = "catalog_oct{oct}.h5"

[output]
format = "hdf5"
path   = "{catdir}"
"""


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    seed, sky = int(sys.argv[1]), os.path.abspath(sys.argv[2])
    cfgdir, catdir = os.path.join(sky, "configs"), os.path.join(sky, "catalogs")
    os.makedirs(cfgdir, exist_ok=True)
    for z in "01":
        for y in "01":
            for x in "01":
                oct_ = z + y + x
                cen = {k: f"{(OBS if b == '1' else -OBS):.4f}" for k, b in (("cenx", x), ("ceny", y), ("cenz", z))}
                path = os.path.join(cfgdir, f"config_oct{oct_}.toml")
                with open(path, "w") as f:
                    f.write(TEMPLATE.format(seed=seed, oct=oct_, ntile=NTILE, nmesh=NMESH, nbuff=NBUFF,
                                            tile=NMESH * CELL, catdir=catdir, **cen))
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
