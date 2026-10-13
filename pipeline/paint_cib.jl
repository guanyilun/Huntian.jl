#!/usr/bin/env julia
# CIB maps of one octant with the XGPaint CIB_Planck2013 model (the Websky CIB model): sources are generated
# once and painted at every requested frequency. Maps are MJy/sr, calibrated to Planck 2013.
#
#   julia --project=$XGPAINT -t N pipeline/paint_cib.jl <OCT> <catalog_AM.h5> <outdir> [nside] [freqs] [tag]
#
# Catalog: finalized AM catalog (Eulerian Mpc/h, km/s); observer from the octZYX bits. Halos are kept above
# Websky's z-dependent completeness mass (data/websky_halo_mass_completion.txt, from the Websky release).
# Pixels are accumulated serially here: XGPaint's HEALPix paint! adds fluxes into shared pixels from
# Threads.@threads (a rare lost-update race), so only its fill_fluxes! is used.
using XGPaint, Healpix, HDF5, Printf, Statistics, DelimitedFiles, Random
import Cosmology: comoving_radial_dist
import Unitful

const hub = 0.68
const rho_mh = 2.775e11 * 0.31
const MMIN_MSUN = 1.0e12                     # CIB model min_mass
# Planck 2013 XXX calibration (545 GHz, ℓ = 411–592, 350 mJy cut), the same factor at every frequency. Flux is
# ∝ 1/shang_I0, so this is the model with shang_I0 = 92/CIB_CALIB = 103.84 instead of XGPaint's default 92.
const CIB_CALIB = 0.8859746653376094

OCT = ARGS[1]; CAT = ARGS[2]; OUTD = ARGS[3]
nside = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4096
freqs = length(ARGS) >= 5 ? parse.(Float64, split(ARGS[5], ",")) : [100.0, 143, 217, 353, 545, 857]
TAG = length(ARGS) >= 6 ? ARGS[6] : "oct$(OCT)"
length(OCT) == 3 && all(in("01"), OCT) || error("OCT must be three 0/1 bits (ZYX)")
obs = ntuple(d -> OCT[4-d] == '1' ? 2618.0 : -2618.0, 3)     # OCT = "ZYX", Mpc/h
mkpath(OUTD)

_mc = readdlm(joinpath(@__DIR__, "..", "data", "websky_halo_mass_completion.txt"); comments=true, comment_char='#')
const MC_Z = Float64.(_mc[:, 1]); const MC_M = Float64.(_mc[:, 2])
function mmin_of_z(z)
    i = clamp(searchsortedlast(MC_Z, z), 1, length(MC_Z) - 1)
    t = clamp((z - MC_Z[i]) / (MC_Z[i+1] - MC_Z[i]), 0.0, 1.0)
    MC_M[i] * (1 - t) + MC_M[i+1] * t
end

cosmo = get_cosmology(Float32; h=Float32(hub), OmegaM=0.31f0)
zg = collect(0.0:0.002:6.0)
chig = [Unitful.ustrip(comoving_radial_dist(cosmo, z)) * hub for z in zg]   # Mpc/h
function chi2z(chi)
    i = clamp(searchsortedlast(chig, chi), 1, length(zg) - 1)
    t = (chi - chig[i]) / (chig[i+1] - chig[i])
    zg[i] * (1 - t) + zg[i+1] * t
end
const CHIMAX = chig[searchsortedfirst(zg, 4.5)]              # z < 4.5, as the other maps

# halo catalog (huntian-catalog/1 HDF5); read with HDF5.jl directly since this runs in the XGPaint environment
function load_halos(path)
    x, y, z, R = h5open(f -> (read(f["x"]), read(f["y"]), read(f["z"]), read(f["RTHL"])), path)
    px = Float32[]; py = Float32[]; pz = Float32[]; mm = Float32[]
    @inbounds for k in eachindex(R)
        M = 4 / 3 * π * rho_mh * Float64(R[k])^3 / hub                  # Msun (M200m proxy)
        M > MMIN_MSUN || continue
        dx = Float64(x[k]) - obs[1]; dy = Float64(y[k]) - obs[2]; dz = Float64(z[k]) - obs[3]
        r = sqrt(dx^2 + dy^2 + dz^2)                                     # Mpc/h
        (30.0 <= r <= CHIMAX) || continue
        M * hub > mmin_of_z(chi2z(r)) || continue
        push!(px, Float32(dx / hub)); push!(py, Float32(dy / hub)); push!(pz, Float32(dz / hub))
        push!(mm, Float32(M))
    end
    pos = Matrix{Float32}(undef, 3, length(mm))
    pos[1, :] = px; pos[2, :] = py; pos[3, :] = pz
    pos, mm
end

# XGPaint draws satellites, luminosity scatter and positions from the default task-local RNG inside
# Threads.@threads :static loops, so seeding it here makes the maps reproducible for a fixed thread count.
# The seed is derived from the sky's seed (catalog provenance) and the octant.
const SKY_SEED = parse(Int, h5open(f -> attrs(f["provenance"])["recipe.seed"], CAT))
const CIB_SEED = 8 * SKY_SEED + parse(Int, OCT; base=2) + 1_000_003
Random.seed!(CIB_SEED)

t0 = time()
@info "paint_cib" OCT obs CAT nside freqs CIB_SEED threads=Threads.nthreads()
halo_pos, halo_mass = load_halos(CAT)
@info "halos loaded" n=length(halo_mass) min=round((time() - t0) / 60; digits=1)
model = CIB_Planck2013{Float32}(nside=nside)
sources = generate_sources(model, cosmo, halo_pos, halo_mass)
halo_pos = nothing; halo_mass = nothing; GC.gc()
@info "sources" N_cen=sources.N_cen N_sat=sources.N_sat min=round((time() - t0) / 60; digits=1)

fc = Vector{Float32}(undef, sources.N_cen); fs = Vector{Float32}(undef, sources.N_sat)
acc = zeros(Float64, 12nside^2)
Ωpix = Healpix.nside2pixarea(nside)
for ν in freqs
    XGPaint.fill_fluxes!(Float32(ν * 1e9), model, sources, fc, fs)
    fill!(acc, 0.0)
    @inbounds for i in 1:sources.N_cen; acc[sources.hp_ind_cen[i]] += fc[i]; end
    @inbounds for i in 1:sources.N_sat; acc[sources.hp_ind_sat[i]] += fs[i]; end
    tot = sum(Float64, fc) + sum(Float64, fs)
    abs(sum(acc) / tot - 1) < 1e-9 || error("flux bookkeeping failed at $ν GHz")
    m = HealpixMap{Float32,RingOrder}(nside); m.pixels .= Float32.(acc .* (CIB_CALIB / Ωpix))
    out = joinpath(OUTD, "cib_nu$(lpad(Int(ν), 4, '0'))_$(TAG)_nside$(nside).fits")
    Healpix.saveToFITS(m, "!" * out, typechar="E")
    @printf("%4.0f GHz: total flux %.4e MJy (uncalibrated), octant mean %.4e MJy/sr -> %s (%.1f min)\n",
            ν, tot, CIB_CALIB * tot / (4π / 8), out, (time() - t0) / 60)
end
