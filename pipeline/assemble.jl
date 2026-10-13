#!/usr/bin/env julia
# Full-sky maps of one sky from its 8 octants. Each octant map is a full-sky HEALPix array that is (almost) zero
# outside its octant; halo and field terms that spill across an octant boundary belong to the sky, so the full
# sky is the plain sum over octants.
#
#   julia --project=pipeline -t 8 pipeline/assemble.jl <skydir> [product ...]
# Reads <skydir>/{fieldmaps,halomaps,cibmaps}/*_octZYX_nside4096.fits and writes <skydir>/maps/:
#   kappa_lt4.5                = field κ + compensated halo κ          (z < 4.5)
#   kappa_field, kappa_halo, kappa_halo_uncompensated                  (components)
#   ksz_total_uK               = T_CMB·field kSZ + compensated halo kSZ [μK]
#   ksz_field_uK, ksz_halo_uK, ksz_halo_uncompensated_uK               (components, μK)
#   tsz_y, isw_uK, cib_nuXXXX (MJy/sr, 100–857 GHz)
# products (default all): kappa ksz tsz isw cib
using Healpix, Printf

const SKY = ARGS[1]
const FM = joinpath(SKY, "fieldmaps"); const HM = joinpath(SKY, "halomaps")
const CM = joinpath(SKY, "cibmaps"); const OUT = joinpath(SKY, "maps")
const OCTS = ["000", "001", "010", "011", "100", "101", "110", "111"]
const TCMB_UK = 2.7255e6
const NSIDE = 4096

readmap(p) = (isfile(p) || error("missing $p"); Healpix.readMapFromFITS(p, 1, Float32))

# Σ_oct Σ_(pattern, scale) scale·map(pattern(oct)) → <OUT>/<name>.fits
function assemble(name, terms)
    acc = zeros(Float64, 12NSIDE^2)
    for oct in OCTS, (pat, scale) in terms
        m = readmap(pat(oct))
        m.resolution.nside == NSIDE || error("nside mismatch in $(pat(oct))")
        acc .+= scale .* m.pixels
    end
    hm = HealpixMap{Float32,RingOrder}(NSIDE); hm.pixels .= Float32.(acc)
    out = joinpath(OUT, "$(name).fits")
    Healpix.saveToFITS(hm, "!" * out, typechar="E")
    @printf("%-28s mean %+.4e  rms %.4e  zero-pix %d -> %s\n", name, sum(acc) / length(acc),
            sqrt(sum(abs2, acc) / length(acc)), count(iszero, acc), out)
end

fld(k) = oct -> joinpath(FM, "$(k)_oct$(oct)_nside$(NSIDE).fits")
hal(k) = oct -> joinpath(HM, "$(k)_oct$(oct)_nside$(NSIDE).fits")

prods = length(ARGS) > 1 ? ARGS[2:end] : ["kappa", "ksz", "tsz", "isw", "cib"]
mkpath(OUT)
if "kappa" in prods
    assemble("kappa_lt4.5", [(fld("kappa"), 1.0), (hal("kappa_halo_comp"), 1.0)])
    assemble("kappa_field", [(fld("kappa"), 1.0)])
    assemble("kappa_halo", [(hal("kappa_halo_comp"), 1.0)])
    assemble("kappa_halo_uncompensated", [(hal("kappa_halo_plain"), 1.0)])
end
if "ksz" in prods
    assemble("ksz_total_uK", [(fld("ksz"), TCMB_UK), (hal("ksz_halo_Wc"), 1.0)])
    assemble("ksz_field_uK", [(fld("ksz"), TCMB_UK)])
    assemble("ksz_halo_uK", [(hal("ksz_halo_Wc"), 1.0)])
    assemble("ksz_halo_uncompensated_uK", [(hal("ksz_halo_W"), 1.0)])
end
"tsz" in prods && assemble("tsz_y", [(hal("tsz_y"), 1.0)])
"isw" in prods && assemble("isw_uK", [(fld("isw"), TCMB_UK)])
if "cib" in prods
    for ν in (100, 143, 217, 353, 545, 857)
        name = "cib_nu$(lpad(ν, 4, '0'))"
        assemble(name, [(oct -> joinpath(CM, "$(name)_oct$(oct)_nside$(NSIDE).fits"), 1.0)])
    end
end
