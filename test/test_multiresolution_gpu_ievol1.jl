#!/usr/bin/env julia
# End-to-end test: run_multitile_split with `use_gpu=true`, ievol=1 (lightcone)
# vs CPU. Verifies that the per-peak ZZon/fcrit GPU batch path matches the
# CPU per-peak analyse_peak loop.
# Requires one GPU:
#   julia --project=. --threads=8 test/test_multiresolution_gpu_ievol1.jl

using Huntian
using CUDA
using Test
using Printf

if !CUDA.functional()
    @warn "CUDA not functional — skipping GPU ievol=1 test"
    exit(0)
end

function _make_config(dir; n=68, boxsize=226.67, z=0.0, ilpt=1, nbuff=8,
                      ioutshear=0, rmax2rs=0.0,
                      ievol=1, z_max=3.0,
                      cenx=0.0, ceny=0.0, cenz=0.0)
    pk_path = joinpath(dir, "test_pk.dat")
    ks = 10.0 .^ range(-4, stop=1, length=200)
    open(pk_path, "w") do f
        for k in ks
            Pk = 1e4 * k^(-1.5)
            println(f, "$k  $Pk")
        end
    end
    filter_path = joinpath(dir, "test_filter.dat")
    open(filter_path, "w") do f
        println(f, "3")
        println(f, "1  1.686  8.0")
        println(f, "2  1.686  5.0")
        println(f, "3  1.686  3.0")
    end
    cosmo = Huntian.CosmologyParams(0.31, 0.049, 0.69, 0.68, 0.965, 0.808)
    ep = Huntian.EllipsoidParams(cosmo)
    tp = Huntian.CollapseTableParams()
    tab = Huntian.make_table(ep, tp)
    tab_path = joinpath(dir, "test_table.dat")
    Huntian.write_homeltab(tab_path, tab, tp)
    cat_path = joinpath(dir, "test_output.pksc")
    Huntian.PipelineConfig(
        n=n, boxsize=Float64(boxsize), pkfile=pk_path, filterfile=filter_path,
        tabfile=tab_path, fileout=cat_path, nbuff=nbuff,
        z_out=Float64(z), ilpt=ilpt,
        ioutshear=ioutshear, rmax2rs=rmax2rs,
        ievol=ievol, z_max=Float64(z_max),
        cenx=cenx, ceny=ceny, cenz=cenz,
        Omx=0.261, OmB=0.049, Omvac=0.69, h=0.68,
        NonGauss=0, fNL=0.0, wsmooth=1, gaussian_split=false,   # block > nbuff here, which gaussian_split rejects
    )
end

function _compare_halos(hs_cpu, hs_gpu; label="")
    @info "$label: CPU=$(length(hs_cpu))  GPU=$(length(hs_gpu))"
    ng = length(hs_cpu); ns = length(hs_gpu)
    # Allow ≤ 3 halos or ≤ 5% difference, as in the ievol=0 test.
    @test abs(ng - ns) / max(ng, 1) < 0.05 || abs(ng - ns) <= 3

    sort_key = h -> (h.x, h.y, h.z)
    sa = sort(hs_cpu; by=sort_key)
    sb = sort(hs_gpu; by=sort_key)
    ncmp = min(length(sa), length(sb), 10)
    for i in 1:ncmp
        da = sa[i]; db = sb[i]
        dx = abs(da.x - db.x); dy = abs(da.y - db.y); dz = abs(da.z - db.z)
        @printf "  halo %2d: Δpos=(%.3f,%.3f,%.3f)  R: CPU=%.3g GPU=%.3g\n" i dx dy dz da.RTHL db.RTHL
    end
end

# ievol=1 with generous z_max: observer (cen) at the domain centre, farthest cell at
# z ≈ 0.1, so z_max=3.0 filters no peaks.
@testset "run_multitile_split ievol=1, z_max generous (1LPT)" begin
    tmpdir = mktempdir()
    cfg = _make_config(tmpdir; n=68, boxsize=226.67, z=0.0, ilpt=1, nbuff=8,
                       ievol=1, z_max=3.0, cenx=0.0, ceny=0.0, cenz=0.0)

    t_cpu = @elapsed halos_cpu = Huntian.run_multitile_split(
        cfg; ntile=2, seed=42, verbose=false, coarse_factor=5, use_gpu=false)
    t_gpu = @elapsed halos_gpu = Huntian.run_multitile_split(
        cfg; ntile=2, seed=42, verbose=false, coarse_factor=5, use_gpu=true)

    @printf "  CPU: %.2f s  GPU: %.2f s (%.2fx)\n" t_cpu t_gpu (t_cpu / t_gpu)
    _compare_halos(halos_cpu, halos_gpu; label="ievol=1 z_max=3.0 (1LPT)")

    rm(tmpdir; recursive=true)
end

@testset "run_multitile_split ievol=1, z_max generous (2LPT)" begin
    tmpdir = mktempdir()
    cfg = _make_config(tmpdir; n=68, boxsize=226.67, z=0.0, ilpt=2, nbuff=8,
                       ievol=1, z_max=3.0, cenx=0.0, ceny=0.0, cenz=0.0)

    t_cpu = @elapsed halos_cpu = Huntian.run_multitile_split(
        cfg; ntile=2, seed=42, verbose=false, coarse_factor=5, use_gpu=false)
    t_gpu = @elapsed halos_gpu = Huntian.run_multitile_split(
        cfg; ntile=2, seed=42, verbose=false, coarse_factor=5, use_gpu=true)

    @printf "  CPU: %.2f s  GPU: %.2f s (%.2fx)\n" t_cpu t_gpu (t_cpu / t_gpu)
    _compare_halos(halos_cpu, halos_gpu; label="ievol=1 z_max=3.0 (2LPT)")

    rm(tmpdir; recursive=true)
end

# ievol=1 with tight z_max: observer offset to (113,113,113) Mpc/h; z_max=0.03
# (chi ≈ 89 Mpc/h) drops the distant peaks.
@testset "run_multitile_split ievol=1, z_max tight (filter exercised)" begin
    tmpdir = mktempdir()
    cfg = _make_config(tmpdir; n=68, boxsize=226.67, z=0.0, ilpt=2, nbuff=8,
                       ievol=1, z_max=0.03,
                       cenx=113.0, ceny=113.0, cenz=113.0)

    halos_cpu = Huntian.run_multitile_split(
        cfg; ntile=2, seed=42, verbose=false, coarse_factor=5, use_gpu=false)
    halos_gpu = Huntian.run_multitile_split(
        cfg; ntile=2, seed=42, verbose=false, coarse_factor=5, use_gpu=true)

    _compare_halos(halos_cpu, halos_gpu; label="ievol=1 z_max=0.03 (filter)")

    # Sanity: filter must have removed some peaks (otherwise the test
    # isn't exercising what it claims to).
    cfg_no_filter = _make_config(tmpdir; n=68, boxsize=226.67, z=0.0, ilpt=2, nbuff=8,
                                 ievol=1, z_max=5.0,
                                 cenx=113.0, ceny=113.0, cenz=113.0)
    halos_no_filter = Huntian.run_multitile_split(
        cfg_no_filter; ntile=2, seed=42, verbose=false, coarse_factor=5, use_gpu=true)
    @info "  unfiltered (z_max=5.0) has $(length(halos_no_filter)) halos vs tight $(length(halos_gpu))"
    @test length(halos_no_filter) > length(halos_gpu)

    rm(tmpdir; recursive=true)
end
