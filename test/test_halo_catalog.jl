using Huntian
using Test
using HDF5
using Random

@testset "HDF5 halo catalog" begin
    rng = MersenneTwister(3)
    tmp = mktempdir()
    prov = Dict("recipe.seed" => "777", "code.commit" => "abc")
    for (T, nf) in ((HaloRecord, 11), (ExtHaloRecord, 33))
        n = 5000
        hs = [T(ntuple(_ -> randn(rng, Float32) * 100, nf)...) for _ in 1:n]
        p = joinpath(tmp, "cat_$(nf).h5")
        write_halo_catalog(p, hs; z_out=0.0, provenance=prov, attributes=Dict("observer" => [-2618.0, 2618.0, -2618.0]),
                           chunk=1000)
        hs2, meta, prov2 = read_halo_catalog(p)
        @test eltype(hs2) === T
        @test hs2 == hs                                        # lossless, every field
        @test meta["n_halos"] == n && meta["observer"] == [-2618.0, 2618.0, -2618.0]
        @test meta["RTHLmax"] == Float64(maximum(h.RTHL for h in hs))
        @test prov2 == prov
        c = read_halo_columns(p, (:x, :RTHL); range=101:250)
        @test c.x == [h.x for h in hs[101:250]] && c.RTHL == [h.RTHL for h in hs[101:250]]
        @test h5open(f -> attrs(f["vx"])["units"], p) == "km/s"
    end
    p = joinpath(tmp, "empty.h5")
    write_halo_catalog(p, HaloRecord[]; z_out=0.0)
    hs2, meta, _ = read_halo_catalog(p)
    @test isempty(hs2) && meta["n_halos"] == 0
    @test_throws Exception read_halo_catalog(joinpath(@__DIR__, "data", "test_pk.dat"))
end
