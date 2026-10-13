@testset "EllipsoidalCollapse" begin

using .Huntian

# Planck 2018
cosmo = CosmologyParams(0.315, 0.049, 0.685, 0.674, 0.965, 0.808)

@testset "Carlson RD" begin
    # RD(x,x,x) = x^{-3/2}
    for x in [1.0, 2.0, 0.5, 10.0]
        rd = Huntian._elliptic_rd(x, x, x)
        @test rd ≈ x^(-1.5) rtol=1e-10
    end

    @test Huntian._elliptic_rd(1.0, 2.0, 3.0) > 0
    @test isfinite(Huntian._elliptic_rd(1.0, 2.0, 3.0))
end

@testset "Shape integrals get_b_2" begin
    # Spherical case: (1,1,1) -> (1/3, 1/3, 1/3)
    b = get_b_2(1.0, 1.0, 1.0, 1)
    @test b[1] ≈ 1/3 rtol=1e-10
    @test b[2] ≈ 1/3 rtol=1e-10
    @test b[3] ≈ 1/3 rtol=1e-10

    # Sum is always 1
    for (a1, a2, a3) in [(0.5, 0.8, 1.2), (0.3, 0.7, 1.5), (0.9, 0.95, 1.0)]
        b = get_b_2(a1, a2, a3, 1)
        @test sum(b) ≈ 1.0 rtol=1e-10
    end

    # For a1 <= a2 <= a3, r_3 = a3/a1 >= 1 always trips the r_3 >= 0.999 check, so
    # b = (1/3, 1/3, 1/3); the Carlson RD branch is only reached for other axis orderings.
    b = get_b_2(0.5, 0.8, 1.2, 1)
    @test b[1] ≈ 1/3 rtol=1e-10
end

@testset "DlinearTables" begin
    tables = Dlinear_tables(cosmo)

    dlin, HD_Ha, D_a = Dlinear_ab(1.0, tables)
    @test dlin ≈ 1.0 rtol=1e-3

    for a in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
        dlin, HD_Ha, D_a = Dlinear_ab(a, tables)
        @test dlin > 0
        @test isfinite(dlin)
    end

    @test Dfnofa(1.0, tables) ≈ 1.0 rtol=1e-3

    # D ∝ a at early times
    d1 = Dfnofa(0.01, tables)
    d2 = Dfnofa(0.02, tables)
    @test d2 / d1 ≈ 2.0 rtol=0.05

    # Compare with CPT92 approximation
    for z in [0.0, 0.5, 1.0, 2.0]
        a = 1.0 / (1.0 + z)
        d_table = Dfnofa(a, tables)
        d_cpt = growth_factor(z, cosmo)
        @test d_table ≈ d_cpt rtol=5e-3
    end
end

@testset "Single ellipsoid: spherical case" begin
    ep = EllipsoidParams(cosmo)

    # e = p = 0 with Frho = 1.686 should collapse just above z = 0
    Frho = 1.686
    zvir, Dvir, fcvir = evolve_ellipse_full(Frho, 0.0, 0.0, ep)
    @test zvir > -0.5
    @test zvir > 0
    @test isfinite(zvir)
end

@testset "Single ellipsoid: known values" begin
    ep = EllipsoidParams(cosmo)

    zvir_high, _, _ = evolve_ellipse_full(5.0, 0.0, 0.0, ep)
    zvir_low, _, _ = evolve_ellipse_full(1.5, 0.0, 0.0, ep)
    @test zvir_high > zvir_low

    zvir_e0, _, _ = evolve_ellipse_full(3.0, 0.0, 0.0, ep)
    zvir_e3, _, _ = evolve_ellipse_full(3.0, 0.3, 0.0, ep)
    @test zvir_e0 > 0
    @test zvir_e3 > 0
    @test isfinite(zvir_e0)
    @test isfinite(zvir_e3)
end

@testset "CollapseTable round-trip" begin
    ep = EllipsoidParams(cosmo)

    tp = CollapseTableParams(Nx=5, Ny=3, Nz=3)
    table = make_table(ep, tp; verbose=false)

    @test size(table) == (5, 3, 3)

    tmppath = tempname()
    write_homeltab(tmppath, table, tp)
    table2, tp2 = read_homeltab(tmppath)
    rm(tmppath)

    @test size(table2) == size(table)
    @test table2 ≈ table

    @test tp2.Nx == tp.Nx
    @test tp2.Ny == tp.Ny
    @test tp2.Nz == tp.Nz
    @test tp2.X1 ≈ tp.X1 rtol=1e-6
    @test tp2.X2 ≈ tp.X2 rtol=1e-6
end

@testset "Trilinear interpolation" begin
    tp = CollapseTableParams(Nx=11, Ny=11, Nz=11,
        X1=0.0, X2=1.0, Y1=0.0, Y2=1.0, Z1=0.0, Z2=1.0)

    # Table = x + y + z: trilinear interpolation is exact for linear functions
    dX = 0.1; dY = 0.1; dZ = 0.1
    table = Float32[
        Float32((i-1)*dX + (j-1)*dY + (k-1)*dZ)
        for i in 1:11, j in 1:11, k in 1:11
    ]

    ct = CollapseTableInterp(table, tp)

    @test Huntian.interpolate(ct, 0.0, 0.0, 0.0) ≈ 0.0 atol=1e-5
    @test Huntian.interpolate(ct, 0.5, 0.0, 0.0) ≈ 0.5 atol=1e-5
    @test Huntian.interpolate(ct, 0.0, 0.5, 0.0) ≈ 0.5 atol=1e-5

    @test Huntian.interpolate(ct, 0.25, 0.25, 0.25) ≈ 0.75 atol=1e-4

    # Out of bounds returns -1
    @test Huntian.interpolate(ct, -0.1, 0.0, 0.0) == -1.0
    @test Huntian.interpolate(ct, 1.1, 0.0, 0.0) == -1.0
end

@testset "OrdinaryDiffEq solver" begin
    using OrdinaryDiffEq

    ep_rk4 = EllipsoidParams(cosmo; solver=:rk4)
    ep_deq = EllipsoidParams(cosmo; solver=:diffeq)

    # Spherical case is sensitive to step size near the singularity, hence 3%
    z_rk4, D_rk4, fc_rk4 = evolve_ellipse_full(1.686, 0.0, 0.0, ep_rk4)
    z_deq, D_deq, fc_deq = evolve_ellipse_full(1.686, 0.0, 0.0, ep_deq)
    @test z_deq ≈ z_rk4 rtol=3e-2
    @test D_deq ≈ D_rk4 rtol=3e-2

    z_rk4, _, _ = evolve_ellipse_full(5.0, 0.0, 0.0, ep_rk4)
    z_deq, _, _ = evolve_ellipse_full(5.0, 0.0, 0.0, ep_deq)
    @test z_deq ≈ z_rk4 rtol=3e-2

    # Triaxial: tighter agreement since axes are distinct
    z_rk4, _, _ = evolve_ellipse_full(3.0, 0.3, 0.0, ep_rk4)
    z_deq, _, _ = evolve_ellipse_full(3.0, 0.3, 0.0, ep_deq)
    @test z_deq ≈ z_rk4 rtol=1e-2

    z_rk4, _, _ = evolve_ellipse_full(5.0, 0.1, 0.05, ep_rk4)
    z_deq, _, _ = evolve_ellipse_full(5.0, 0.1, 0.05, ep_deq)
    @test z_deq ≈ z_rk4 rtol=1e-2

    # No collapse
    z_rk4, _, _ = evolve_ellipse_full(0.5, 0.0, 0.0, ep_rk4)
    z_deq, _, _ = evolve_ellipse_full(0.5, 0.0, 0.0, ep_deq)
    @test z_rk4 == -1.0
    @test z_deq == -1.0

    z_high, _, _ = evolve_ellipse_full(5.0, 0.0, 0.0, ep_deq)
    z_low, _, _ = evolve_ellipse_full(1.5, 0.0, 0.0, ep_deq)
    @test z_high > z_low
end

end # EllipsoidalCollapse testset
