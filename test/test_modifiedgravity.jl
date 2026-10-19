using Random, FFTW
@testset "ModifiedGravity" begin
    c = CosmologyParams(0.281, 0.046, 0.719, 0.697, 0.971, 0.82)   # ELEPHANT (WMAP9)

    @testset "GR limit" begin
        t = mg_growth_table(GR(), c; nk=9, na=11)
        @test all(t.RD .≈ 1) && all(t.Rv .≈ 1)
    end

    @testset "f(R) Compton scale" begin
        # λ_C = 1/(a m) ≈ 32 √(|f_R0|/1e-4) Mpc today (order of magnitude, Hu & Sawicki n=1)
        lam = 1 / compton_k(HuSawicki(fR0=1e-4), 1.0, c) / c.h
        @test 20 < lam < 40
        @test compton_k(HuSawicki(fR0=1e-6), 1.0, c) ≈ 10 * compton_k(HuSawicki(fR0=1e-4), 1.0, c) rtol=1e-12
        @test mu(HuSawicki(fR0=1e-5), 1e-5, 1.0, c) ≈ 1 atol=1e-5
        @test mu(HuSawicki(fR0=1e-5), 1e4, 1.0, c) ≈ 4/3 atol=1e-5
    end

    @testset "f(R) growth limits" begin
        t = mg_growth_table(HuSawicki(fR0=1e-5), c; kmin=1e-4, kmax=1e2, nk=41, na=19)
        @test growth_ratio(t, 1e-4, 1.0) ≈ 1 atol=1e-4           # GR on large scales
        @test growth_ratio(t, 10.0, 1.0) > growth_ratio(t, 0.1, 1.0) > 1
        @test velocity_ratio(t, 10.0, 1.0) > growth_ratio(t, 10.0, 1.0)   # f grows too
        @test growth_ratio(t, 1.0, 0.5) < growth_ratio(t, 1.0, 1.0)
    end

    @testset "Matter-era growing mode with μ = 4/3" begin
        # Constant μ=4/3 and Ω_m≈1 give D ∝ a^p, p = (√33 − 1)/4.
        cm = CosmologyParams(1.0, 0.046, 0.0, 0.7, 0.97, 0.8)
        D, _ = Huntian.ModifiedGravity._solve_growth(a -> 4/3, [1e-2], cm)
        @test D[1] / 1e-3 ≈ 10^((sqrt(33) - 1) / 4) rtol=1e-6
    end

    @testset "nDGP scale independence" begin
        t = mg_growth_table(NDGP(H0rc=1.0), c; nk=9, na=11)
        @test maximum(t.RD[:, end]) - minimum(t.RD[:, end]) < 1e-12
        @test 1.03 < growth_ratio(t, 0.1, 1.0) < 1.2
    end

    @testset "Yukawa range" begin
        t = mg_growth_table(Yukawa(eps=0.3, lambda=5.0), c; nk=41, na=19)
        @test growth_ratio(t, 1e-3, 1.0) ≈ 1 atol=1e-3
        # k ≫ 1/λ: same as the scale-independent μ = 1 + ε(a), solved directly
        MG = Huntian.ModifiedGravity
        D, _ = MG._solve_growth(a -> 1 + 0.3 * c.OL / MG._Em2(a, c) / c.OL, [1.0], c)
        Dg, _ = MG._solve_growth(a -> 1.0, [1.0], c)
        @test growth_ratio(t, 100.0, 1.0) ≈ D[1] / Dg[1] rtol=1e-4
    end
end

@testset "MG patch corrections" begin
    c = CosmologyParams(0.281, 0.046, 0.719, 0.697, 0.971, 0.82)
    n, L = 64, 128.0
    dx = L / n
    delta = randn(MersenneTwister(1), n, n, n)
    xq = [dx * (I - (n + 1) / 2) for I in (20, 33, 47)] |> v -> [v[1] v[2] v[3]; v[2] v[3] v[1]; v[3] v[1] v[2]]
    R = [8.0, 10.0, 12.0]

    tg = mg_growth_table(GR(), c; nk=9, na=11)
    dD, dv = mg_patch_corrections(delta, L, xq, R, 1.0, tg)
    @test all(iszero, dD) && all(iszero, dv)

    # nDGP: scale-independent R, so the correction is (R − 1) × the patch mean of ψ.
    tn = mg_growth_table(NDGP(H0rc=1.0), c; nk=9, na=11)
    dD, dv = mg_patch_corrections(delta, L, xq, R, 1.0, tn; nR=3)
    psi = displacements_1lpt(rfft(delta), n, L)
    for h in 1:3
        I = round.(Int, xq[:, h] ./ dx .+ (n + 1) / 2)
        r = ceil(Int, R[h] / dx); m = zeros(3); cnt = 0
        for i in -r:r, j in -r:r, k in -r:r
            (i^2 + j^2 + k^2) * dx^2 <= R[h]^2 || continue
            ii, jj, kk = mod1.(I .+ (i, j, k), n); cnt += 1
            m .+= (psi[1][ii, jj, kk], psi[2][ii, jj, kk], psi[3][ii, jj, kk])
        end
        m ./= cnt
        @test dD[:, h] ≈ (growth_ratio(tn, 0.1, 1.0) - 1) .* m rtol=0.05
        @test dv[:, h] ≈ (velocity_ratio(tn, 0.1, 1.0) - 1) .* m rtol=0.05
    end
end

@testset "finalize_eulerian MG corrections" begin
    c = CosmologyParams(0.281, 0.046, 0.719, 0.697, 0.971, 0.82)
    h = HaloRecord(1f0, 2f0, 3f0, 0.5f0, -0.2f0, 0.1f0, 2f0, 0.01f0, 0f0, -0.01f0, 1.5f0)
    base = finalize_eulerian([h], c, (0.0, 0.0, 0.0); ievol=0, z_out=0.0)[1]
    dD = reshape([0.1, 0.0, 0.0], 3, 1); dv = reshape([0.0, 0.2, 0.0], 3, 1)
    mg = finalize_eulerian([h], c, (0.0, 0.0, 0.0); ievol=0, z_out=0.0, ddisp=dD, dvel=dv)[1]
    @test mg.x ≈ base.x + 0.1f0 && mg.y == base.y && mg.vx == base.vx
    @test mg.vy - base.vy ≈ 0.2 * (base.vx / (0.5 + 2 * 0.01)) rtol=1e-4   # vfac · dv
end

@testset "MG 2LPT corrections" begin
    MG = Huntian.ModifiedGravity
    c = CosmologyParams(0.281, 0.046, 0.719, 0.697, 0.971, 0.82)

    # GR second-order growth reproduces Huntian's −(3/7)Ω_m(a)^(−1/143) D²
    D2 = MG._solve_growth2(a -> 1.0, [1.0], c)[1]
    D1, _ = MG._solve_growth(a -> 1.0, [1.0], c)
    @test D2 / D1[1]^2 ≈ -3 / 7 * Huntian.Cosmology.omega_m_a(c.Om, c.OL, 1.0)^(-1 / 143) rtol=2e-3
    r2, rd2 = second_order_growth_ratio(NDGP(H0rc=1.0), 0.1, 1.0, c)
    @test r2 > 1 && abs(r2 / rd2 - 1) < 0.02

    n, L = 48, 96.0
    dx = L / n
    delta = randn(MersenneTwister(2), n, n, n)
    dk = rfft(delta)
    one_(k) = 1.0
    # On a field without Nyquist modes (where the treatments differ) the source gives the same ψ₂ as
    # displacements_2lpt, the path the halo finder uses, and has zero mean (each 2×2 minor averages to zero).
    bl = copy(dk); bl[end, :, :] .= 0; bl[:, n ÷ 2 + 1, :] .= 0; bl[:, :, n ÷ 2 + 1] .= 0
    src = MG._src2_pair_k(bl, n, L, one_, one_)
    psi2_ref = displacements_2lpt(bl, n, L)
    for d in 1:3
        @test displacement_2lpt_component(src, n, L, d) ≈ psi2_ref[d] rtol=1e-8
    end
    @test abs(src[1, 1, 1]) < 1e-10 * maximum(abs, src)

    t = mg_growth_table(NDGP(H0rc=1.0), c; nk=9, na=11)
    rd, rv = growth_ratio(t, 0.1, 1.0), velocity_ratio(t, 0.1, 1.0)
    xq = reshape([dx * (I - (n + 1) / 2) for I in (15, 30, 22)], 3, 1); R = [7.0]
    # nDGP is scale independent, so the 2LPT corrections are exactly (R_D² − 1) D₂ and 2(R_D R_v − 1) D₂ times
    # the patch mean of the GR ψ₂ (the top-hat approximation of the patch is tested at first order above;
    # at second order a white-noise field puts too much power near the grid scale for a cell-sum check).
    delta_bl = irfft(bl, n)
    d1, v1 = mg_patch_corrections(delta_bl, L, xq, R, 1.0, t)
    d2, v2 = mg_patch_corrections(delta_bl, L, xq, R, 1.0, t; D2=-0.4)
    u = xq ./ dx .+ (n - 1) / 2
    P = MG._patch_mean(MG._src2_pair_k(rfft(delta_bl), n, L, one_, one_), one_, -1, n, L, u, R, R)
    @test vec(d2 .- d1) ≈ -0.4 * (rd^2 - 1) .* vec(P) rtol=1e-8
    @test vec(v2 .- v1) ≈ 2 * -0.4 * (rd * rv - 1) .* vec(P) rtol=1e-8
end

@testset "Screening" begin
    MG = Huntian.ModifiedGravity
    c = CosmologyParams(0.281, 0.046, 0.719, 0.697, 0.971, 0.82)

    # A body feeling the field's μ follows the field
    m = HuSawicki(fR0=1e-5)
    D, dD, Db, dDb = MG._solve_growth_body(a -> mu(m, 1.0, a, c), a -> mu(m, 1.0, a, c), [0.5, 1.0], c)
    @test Db ≈ D rtol=1e-12
    @test dDb ≈ dD rtol=1e-12

    t = mg_growth_table(m, c; nk=41, na=19)
    @test growth_ratio(t, 1e-4, 1.0; screened=true) ≈ 1 atol=1e-4
    @test 1 < velocity_ratio(t, 1.0, 1.0; screened=true) < velocity_ratio(t, 1.0, 1.0)   # falls in the MG field, feels no fifth force
    @test 1 < growth_ratio(t, 1.0, 1.0; screened=true) < growth_ratio(t, 1.0, 1.0)
    tg = mg_growth_table(GR(), c; nk=9, na=11)
    @test all(tg.RD0 .≈ 1) && all(tg.Rv0 .≈ 1)

    # Thin-shell charge
    @test fR_background(m, 1.0, c) ≈ 1e-5
    @test fR_background(m, 0.5, c) < 1e-5
    rho = Huntian.ModifiedGravity.RHO_CRIT * c.Om
    R_of(M) = cbrt(3M / (4π * rho))
    Ms = screening_mass(m, 1.0, c)
    @test scalar_charge(m, 0.5Ms, R_of(0.5Ms), 1.0, c) == 1.0
    @test scalar_charge(m, Ms, R_of(Ms), 1.0, c) ≈ 1.0 rtol=1e-10
    @test scalar_charge(m, 8Ms, R_of(8Ms), 1.0, c) ≈ 8^(-2/3) rtol=1e-10      # Φ_N ∝ M^(2/3)
    @test screening_mass(HuSawicki(fR0=1e-6), 1.0, c) ≈ Ms * 10^-1.5 rtol=1e-10
    @test scalar_charge(NDGP(H0rc=1.0), 1e16, R_of(1e16), 1.0, c) == 1.0
    @test isinf(screening_mass(NDGP(H0rc=1.0), 1.0, c))
    @test scalar_charge(Yukawa(eps=0.3, lambda=5.0, phi_s=1e-6), 1e15, R_of(1e15), 1.0, c) < 1

    # Screened corrections use the screened ratios; GR stays zero
    n, L = 32, 64.0
    delta = randn(MersenneTwister(3), n, n, n)
    xq = reshape([0.0, 2.0, -4.0], 3, 1)
    d0, v0 = mg_patch_corrections(delta, L, xq, [5.0], 1.0, tg; screened=true)
    @test all(iszero, d0) && all(iszero, v0)
    tn = mg_growth_table(NDGP(H0rc=1.0), c; nk=9, na=11)
    d1, v1 = mg_patch_corrections(delta, L, xq, [5.0], 1.0, tn)
    ds, vs = mg_patch_corrections(delta, L, xq, [5.0], 1.0, tn; screened=true)
    @test vec(ds) ≈ (growth_ratio(tn, 0.1, 1.0; screened=true) - 1) / (growth_ratio(tn, 0.1, 1.0) - 1) .* vec(d1) rtol=1e-8
    @test vec(vs) ≈ (velocity_ratio(tn, 0.1, 1.0; screened=true) - 1) / (velocity_ratio(tn, 0.1, 1.0) - 1) .* vec(v1) rtol=1e-8
end
