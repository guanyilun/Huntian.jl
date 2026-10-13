#!/usr/bin/env julia
# Full-matter field maps of one octant (every lattice cell, halo + field, z < z_max): κ, mass, τ, kSZ, ISW
# on HEALPix (RING), pixelized on the GPU. Halo-independent, so it can run alongside catalog.jl.
#
#   julia --project=pipeline -t 16 pipeline/fieldmap.jl <config.toml> [nside=4096] [outdir=.]
# Outputs <kernel>_<tag>_nside<nside>.fits with tag = config file name without "config_" and ".toml".
using TOML, Huntian, CUDA, Healpix, Printf, Statistics

function main()
    config_path = ARGS[1]
    nside = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4096
    outdir = length(ARGS) >= 3 ? ARGS[3] : "."
    isdir(outdir) || mkpath(outdir)

    config = TOML.parsefile(config_path)
    cfg = PipelineConfig(config)
    run_cfg = get(config, "run", Dict{String,Any}())
    seed = get(run_cfg, "seed", 42)
    ntile = get(run_cfg, "ntile", 16)
    coarse_factor = get(run_cfg, "coarse_factor", 4)
    tag = replace(basename(config_path), "config_" => "", ".toml" => "")

    @assert CUDA.functional() "CUDA is not functional on this node"
    devices = collect(0:length(CUDA.devices())-1)
    res = Resolution(nside)
    npix = nside2npix(nside)
    v2p = (x, y, z) -> Healpix.vec2pixRing(res, x, y, z)

    @info "fieldmap octant" tag nside npix devices seed ntile coarse_factor
    t0 = time()
    maps = run_multitile_fieldmap(cfg; ntile=ntile, seed=seed, coarse_factor=coarse_factor,
                                  npix=npix, vec2pix=v2p,
                                  kernels=[:kappa, :mass, :tau, :ksz, :isw],
                                  subdiv_max=5,         # CMB source plane: chi_cmb of the run cosmology
                                  isw_mode=:gather,     # ISW as a line-of-sight integral (no deposit floor)
                                  exclude_halos=nothing,
                                  use_gpu=true, devices=devices,
                                  gpu_paint=true, nside=nside,
                                  verbose=true)
    elapsed = time() - t0
    @info "painting complete" elapsed_min = round(elapsed / 60; digits=1)

    mtot = sum(maps[:mass])
    kmean = mean(maps[:kappa])
    kon = count(>(0), maps[:mass]) / npix
    @printf("total painted mass: %.6e Msun/h\n", mtot)
    @printf("mean kappa (full sky, unsubtracted): %.6e\n", kmean)
    @printf("sky fraction covered: %.4f (octant = 0.125)\n", kon)

    for (kern, m) in maps
        hm = HealpixMap{Float64,RingOrder}(m)
        path = joinpath(outdir, "$(kern)_$(tag)_nside$(nside).fits")
        Healpix.saveToFITS(hm, "!" * path, typechar="E")
        @info "wrote $path"
    end
end

main()
