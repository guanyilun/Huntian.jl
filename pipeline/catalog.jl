#!/usr/bin/env julia
# One lightcone octant catalog on the GPUs visible to this process: halo finding (multi-resolution
# tiles), Lagrangian exclusion + volume reduction, Eulerian positions and km/s velocities. Writes an HDF5
# halo catalog (write_halo_catalog) with the provenance embedded.
#
#   julia --project=pipeline -t 32 pipeline/catalog.jl <config.toml>
using TOML
using Huntian
using CUDA
using HDF5

function main()
    if isempty(ARGS)
        println(stderr, "Usage: julia --project=pipeline catalog.jl <config.toml>")
        exit(1)
    end

    config_path = ARGS[1]
    if !isfile(config_path)
        println(stderr, "Error: config not found: $config_path")
        exit(1)
    end

    config = TOML.parsefile(config_path)
    cfg = PipelineConfig(config)
    # captured first so a dirty code tree fails before any GPU time is spent (HUNTIAN_ALLOW_DIRTY=1 embeds the diff)
    prov = capture_provenance(config_path; scripts=[@__FILE__])

    run_cfg = get(config, "run", Dict{String,Any}())
    seed       = get(run_cfg, "seed", 42)
    ntile      = get(run_cfg, "ntile", 16)
    coarse_factor = get(run_cfg, "coarse_factor", 4)
    gen_table  = get(run_cfg, "generate_table", false)
    ode_solver = Symbol(get(run_cfg, "ode_solver", "rk4"))
    vol_red    = Bool(get(run_cfg, "volume_reduction", false))

    out_cfg = get(config, "output", Dict{String,Any}())
    outdir  = get(out_cfg, "path", ".")
    isdir(outdir) || mkpath(outdir)

    if gen_table
        Om_total = cfg.Omx + cfg.OmB
        cosmo_tab = CosmologyParams(Om_total, cfg.OmB, cfg.Omvac, cfg.h, 0.965, 0.808)
        ep = EllipsoidParams(cosmo_tab; solver=ode_solver)
        tp = CollapseTableParams()
        @info "Generating collapse table (solver=$ode_solver)..."
        table = make_table_threaded(ep, tp; verbose=true)
        write_homeltab(cfg.tabfile, table, tp)
        @info "Wrote collapse table: $(cfg.tabfile)"
    end

    @assert CUDA.functional() "CUDA is not functional on this node"
    ndev = length(CUDA.devices())
    ndev < 1 && error("No CUDA devices found")
    devices = collect(0:ndev-1)
    @info "Using $ndev GPUs: $devices"
    prov["env.gpu"] = join(unique(CUDA.name(d) for d in CUDA.devices()), ", ") * " × $ndev"
    prov["env.cuda_runtime"] = string(CUDA.runtime_version())
    prov["env.cuda_driver"] = string(CUDA.driver_version())

    nmesh = cfg.n
    nbuff = cfg.nbuff
    nsub, N = grid_layout(cfg, ntile)
    alatt = cfg.boxsize / nmesh
    boxsize_full = N * alatt

    @info "octant catalog" N=N ntile=ntile nmesh=nmesh nbuff=nbuff nsub=nsub boxsize_full=round(boxsize_full; digits=1) periodic_cores=cfg.periodic_cores seed=seed coarse_factor=coarse_factor ievol=cfg.ievol z_max=cfg.z_max ilpt=cfg.ilpt ioutshear=cfg.ioutshear peak_threshold_per_tile=cfg.peak_threshold_per_tile volume_reduction=vol_red

    t0 = time()
    halos = run_multitile_split(cfg;
        ntile=ntile,
        seed=seed,
        coarse_factor=coarse_factor,
        use_gpu=true,
        devices=devices,
        verbose=true,
        profile=true)
    elapsed = time() - t0

    @info "Pipeline complete" halos=length(halos) elapsed_min=round(elapsed/60; digits=1)

    if length(halos) > 1
        @info "Running merger" exclusion=true volume_reduction=vol_red
        halos = merge_catalog(halos; verbose=true, volume_reduction=vol_red)
        @info "After merge: $(length(halos)) halos"
    end

    # Eulerian positions + km/s peculiar velocities (the Fortran merge_pkvd conversion)
    if !isempty(halos)
        @info "Finalizing (Eulerian position + km/s velocity)..."
        cosmo_fin = CosmologyParams(cfg.Omx + cfg.OmB, cfg.OmB, cfg.Omvac, cfg.h, 0.965, 0.808)
        obs_fin = (Float64(cfg.cenx), Float64(cfg.ceny), Float64(cfg.cenz))
        halos = finalize_eulerian(halos, cosmo_fin, obs_fin; ievol=cfg.ievol, z_out=Float64(cfg.z_out))
    end

    path = joinpath(outdir, replace(basename(cfg.fileout), r"\.\w+$" => "") * ".h5")
    prov["run.n_halos"] = string(length(halos))
    prov["run.pipeline_minutes"] = string(round(elapsed / 60; digits=2))
    prov["run.product"] = "raw catalog (merged + finalized; Eulerian positions, km/s velocities; masses before AM)"
    write_halo_catalog(path, halos; z_out=cfg.z_out, provenance=prov,
                       attributes=Dict("observer" => [cfg.cenx, cfg.ceny, cfg.cenz], "stage" => "raw"))
    @info "Wrote catalog: $path ($(length(halos)) halos)"

    @info "Done: $(length(halos)) halos in $(round(elapsed/60; digits=1)) minutes"
end

main()
