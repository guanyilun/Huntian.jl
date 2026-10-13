using Huntian
using Test
using Random
import Healpix
import Huntian.Cosmology: CosmologyParams, build_chi_to_z, chi_to_z, chi

# CPU checks of run_multitile_fieldmap on a small lightcone box:
# exact bookkeeping identities plus analytic anchors for the mean κ and τ.

@testset "FieldMap" begin
    datadir = joinpath(@__DIR__, "..", "data")
    alatt = 8.0; nmesh = 64; nbuff = 8; ntile = 2
    nsub = nmesh - 2nbuff; N = nsub * ntile + 2nbuff; Nc = nsub * ntile      # N = 112
    obs = (-384.0, -384.0, -384.0)                                            # grid corner
    config = Dict{String,Any}(
        "cosmology" => Dict{String,Any}("Om" => 0.31, "OB" => 0.049, "OL" => 0.69, "h" => 0.68),
        "grid" => Dict{String,Any}("n" => nmesh, "boxsize" => nmesh * alatt, "nbuff" => nbuff,
                                   "cenx" => obs[1], "ceny" => obs[2], "cenz" => obs[3]),
        "run" => Dict{String,Any}("ievol" => 1, "z_max" => 0.1, "z_out" => 0.0, "ilpt" => 2,
                                  "ioutshear" => 0, "gaussian_split" => false),   # gaussian_split tested below
        "files" => Dict{String,Any}("pk" => joinpath(datadir, "pk_websky.dat"),
                                    "filterbank" => joinpath(datadir, "filters_websky.dat"),
                                    "homeltab" => joinpath(datadir, "HomelTab_websky.dat"),
                                    "output" => joinpath(mktempdir(), "fieldmap_test.pksc")),
    )
    cfg = PipelineConfig(config)
    nside = 64; res = Healpix.Resolution(nside); npix = Healpix.nside2npix(nside)
    v2p = (x, y, z) -> Healpix.vec2pixRing(res, x, y, z)
    cosmo = CosmologyParams(0.31, 0.049, 0.69, 0.68, 0.965, 0.808)
    chimax = chi(0.1, cosmo); rho_m = 2.775e11 * 0.31; rmin_eff = 2 * alatt
    cellpos(i) = (i - (Nc + 1) / 2) * alatt
    rdist(x, y, z) = sqrt((x - obs[1])^2 + (y - obs[2])^2 + (z - obs[3])^2)

    @testset "ang2pix_ring (GPU painter) == Healpix.jl" begin
        rng = MersenneTwister(7); nbad = 0
        for _ in 1:200_000
            x, y, z = randn(rng), randn(rng), randn(rng)
            Huntian.MultiResolution.ang2pix_ring(nside, x, y, z) == Healpix.vec2pixRing(res, x, y, z) || (nbad += 1)
        end
        @test nbad == 0
    end

    K = [:kappa, :mass, :tau, :ksz, :isw]
    run(; kw...) = run_multitile_fieldmap(cfg; ntile=ntile, seed=12345, npix=npix, vec2pix=v2p, nside=nside,
                                          use_gpu=false, verbose=false, kw...)
    maps  = run(; kernels=K, subdiv_max=1)
    maps3 = run(; kernels=K, subdiv_max=3)

    @testset "mass conservation (exact cell bookkeeping)" begin
        ncell = count(rmin_eff <= rdist(cellpos(i), cellpos(j), cellpos(k)) <= chimax
                      for k in 1:Nc, j in 1:Nc, i in 1:Nc)
        @test sum(maps[:mass]) ≈ rho_m * alatt^3 * ncell rtol=1e-6
        # sub-cell splitting truncates boundary sub-cells: ~1% low, never more
        @test 0.98 < sum(maps3[:mass]) / sum(maps[:mass]) <= 1.0
    end

    @testset "mean κ and τ vs analytic line-of-sight integrals" begin
        chi2z = build_chi_to_z(cosmo; z_max=1.0); chistar = chi(1089.0, cosmo)
        nint = 2000; dr = (chimax - rmin_eff) / nint; kacc = 0.0; tacc = 0.0
        sigT_ne0 = 6.65246e-29 * 11.2299 * 0.9 * 0.049 * 0.68^2 * 3.0857e22 / 0.68
        for ii in 0:nint-1
            r = rmin_eff + (ii + 0.5) * dr; z = chi_to_z(chi2z, r)
            kacc += 1.5 * 0.31 * (1 / 2997.92458)^2 * (1 + z) * (1 - r / chistar) * r * dr
            x_e = z < 3 ? (1 - 0.245 / 2) : (1 - 3 * 0.245 / 4)
            tacc += sigT_ne0 * x_e * (1 + z)^2 * dr
        end
        # octant of sky; the painted map is grid-quantized at the χ boundaries
        @test sum(maps[:kappa]) / npix ≈ kacc / 8 rtol=0.005
        @test sum(maps[:tau]) / npix ≈ tacc / 8 rtol=0.005
    end

    @testset "kSZ / ISW sanity" begin
        cov = findall(!iszero, maps[:tau])
        kszm = sum(maps[:ksz][cov]) / length(cov); kszr = sqrt(sum(abs2, maps[:ksz][cov]) / length(cov))
        @test abs(kszm) / kszr < 0.5          # signed map: mean ≪ rms
        iswr = sqrt(sum(abs2, maps[:isw][cov]) / length(cov))
        @test isfinite(iswr) && iswr > 0
    end

    @testset "octant containment" begin
        on = findall(>(0), maps[:mass])
        bad = count(p -> any(<(-0.15), Healpix.pix2vecRing(res, p)), on)
        @test bad <= 0.001 * length(on)       # boundary pixels only
        @test 0.10 < length(on) / npix < 0.15 # ≈ 1/8 of the sky
    end

    @testset "multi-worker dispatch == single worker" begin
        # Painted totals are ψ-independent → sums must agree to summation roundoff.
        # Guards against workers aliasing one accumulator (which over-paints by 2^(n-1)).
        maps4 = run(; kernels=K, subdiv_max=3, cpu_workers=4)
        for k in (:mass, :kappa, :isw)
            @test sum(maps4[k]) ≈ sum(maps3[k]) rtol=1e-9
        end
    end

    @testset "exclude_halos removes exactly the enclosed cells" begin
        halos = (x=[-250.0, -150.0], y=[-250.0, -320.0], z=[-250.0, -300.0], R=[30.0, 20.0])
        mex = run(; kernels=[:mass], subdiv_max=1, exclude_halos=halos)
        nex = 0
        for n in 1:2, k in 1:Nc, j in 1:Nc, i in 1:Nc
            x, y, z = cellpos(i), cellpos(j), cellpos(k)
            (x - halos.x[n])^2 + (y - halos.y[n])^2 + (z - halos.z[n])^2 <= halos.R[n]^2 || continue
            rmin_eff <= rdist(x, y, z) <= chimax && (nex += 1)
        end
        @test nex > 0
        @test sum(maps[:mass]) - sum(mex[:mass]) ≈ rho_m * alatt^3 * nex rtol=1e-6
    end

    @testset "periodic_cores: a box-corner observer sees every cell" begin
        # Default layout N = nsub*ntile + 2nbuff: the cores span ±Nc·a/2 but the periodic box
        # is 2nbuff cells wider, so an observer at the BOX corner (the octant-lightcone
        # geometry) misses a slab next to each plane.
        # periodic_cores=true (N = nsub*ntile) makes the box corner the core corner.
        mkcfg(periodic, o) = PipelineConfig(merge(config, Dict{String,Any}(
            "grid" => merge(config["grid"], Dict{String,Any}("periodic_cores" => periodic,
                                                             "cenx" => o, "ceny" => o, "cenz" => o)))))
        for periodic in (false, true)
            c0 = mkcfg(periodic, 0.0)
            nsub_, Nbox = grid_layout(c0, ntile)
            @test nsub_ == nsub && Nbox == (periodic ? nsub * ntile : nsub * ntile + 2nbuff)
            o = -Nbox * alatt / 2                                   # box corner
            c = mkcfg(periodic, o)
            run_fm() = run_multitile_fieldmap(c; ntile=ntile, seed=12345, npix=npix, vec2pix=v2p,
                                              use_gpu=false, verbose=false, kernels=[:mass], subdiv_max=1)
            m = periodic ? run_fm() :
                (@test_logs (:warn, r"outside the tile-core") match_mode=:any run_fm())
            bpos(i) = (i - (Nbox + 1) / 2) * alatt                  # every cell of the box
            dist(x, y, z) = sqrt((x - o)^2 + (y - o)^2 + (z - o)^2)
            nfull = count(rmin_eff <= dist(bpos(i), bpos(j), bpos(k)) <= chimax
                          for k in 1:Nbox, j in 1:Nbox, i in 1:Nbox)
            if periodic
                @test sum(m[:mass]) ≈ rho_m * alatt^3 * nfull rtol=1e-6
            else
                @test sum(m[:mass]) < 0.97 * rho_m * alatt^3 * nfull   # the unsimulated slab
            end
        end
        @test_throws ErrorException Huntian.MultiTile.run_multitile(
            PipelineConfig(merge(config, Dict{String,Any}("grid" => merge(config["grid"],
                Dict{String,Any}("periodic_cores" => true))))); ntile=ntile, seed=1)
    end

    @testset "gaussian_split field maps" begin
        # N = 112 → coarse grid 14 (block 8 ≤ nbuff, r_s = 4); compare with the original splice at the same M
        cfg_g = PipelineConfig(merge(config, Dict("run" => merge(config["run"], Dict{String,Any}("gaussian_split" => true)))))
        mo = run(; kernels=K, subdiv_max=1, coarse_grid=14)
        mg = run_multitile_fieldmap(cfg_g; ntile=ntile, seed=12345, npix=npix, vec2pix=v2p, nside=nside, use_gpu=false,
                                    verbose=false, kernels=K, subdiv_max=1, coarse_grid=14)
        @test sum(mg[:mass]) ≈ sum(mo[:mass]) rtol=1e-6          # cell bookkeeping is field-independent
        cov = findall(!iszero, mo[:mass])
        c(a, b) = (x = a[cov] .- sum(a[cov]) / length(cov); y = b[cov] .- sum(b[cov]) / length(cov); sum(x .* y) / sqrt(sum(abs2, x) * sum(abs2, y)))
        # κ near the observer (z < 0.1, 8 Mpc/h cells) is dominated by a few displaced cells: even the original
        # splice at coarse grids 14/28/56 correlates only 0.38–0.44 with itself, so only ISW is compared here
        @test c(mg[:isw], mo[:isw]) > 0.95
        @test abs(sum(mg[:kappa]) / sum(mo[:kappa]) - 1) < 0.01
        @test all(isfinite, mg[:ksz])
    end

    @testset "ISW gather: pix2vec_ring and exact ray partition over tile cores" begin
        MR = Huntian.MultiResolution
        rng = MersenneTwister(11)
        for ns in (1, 2, 8, 64), _ in 1:2000
            p = rand(rng, 1:12ns^2)
            v = MR.pix2vec_ring(ns, p)
            @test MR.ang2pix_ring(ns, v...) == p
            @test all(isapprox.(v, Healpix.pix2vecRing(Healpix.Resolution(ns), p); atol=1e-12))
        end
        # unit potential & unit kernel on 2×2×2 tiles (core 12 cells, nbuff 2) from the corner:
        # every ray must collect exactly h·#{j : rmin ≤ (j−½)h ≤ χmax} (no gaps, no double counts)
        a = 1.0; nm = 16; nb = 2; nc = nm - 2nb; o = (0.0, 0.0, 0.0); ns = 16
        chimax = 20.0; rmin = 2.0; h = 0.7; nrt = 64; wl = ones(nrt); inv_dr = (nrt - 1) / chimax
        m = zeros(12ns^2); pot = ones(Float32, nm, nm, nm)
        for kt in 0:1, jt in 0:1, it in 0:1
            MR._gather_tile_isw!(m, pot, wl, ns, nm, nb, a, nc * (it + 0.5), nc * (jt + 0.5), nc * (kt + 0.5), o,
                                 rmin, chimax, inv_dr, nrt, h)
        end
        expect = h * count(j -> rmin <= (j - 0.5) * h <= chimax, 1:100)
        inoct = [p for p in 1:12ns^2 if all(>(1e-3), MR.pix2vec_ring(ns, p))]
        @test all(p -> isapprox(m[p], expect; rtol=1e-12), inoct)
        @test count(!iszero, m) <= length(inoct) + 8ns     # only octant pixels (+ face-grazing rays)
        # rays inside a face plane (here the equator, n_z = 0) belong to exactly one of the two tiles sharing it,
        # also when the observer sits on the face up to rounding (the octant layout: neighbouring octants are separate
        # runs with the observer on opposite faces; the configs' 5-decimal tile box leaves ~2e-5 Mpc/h offsets)
        mA = zeros(12ns^2); mB = zeros(12ns^2); oe = (0.0, 0.0, -2e-5)   # production: faces ~2e-5 Mpc/h off
        MR._gather_tile_isw!(mA, pot, wl, ns, nm, nb, a, nc / 2, nc / 2, nc / 2, oe, rmin, chimax, inv_dr, nrt, h)
        MR._gather_tile_isw!(mB, pot, wl, ns, nm, nb, a, nc / 2, nc / 2, -nc / 2, oe, rmin, chimax, inv_dr, nrt, h)
        eq = [p for p in 1:12ns^2 if (v = MR.pix2vec_ring(ns, p); abs(v[3]) < 1e-12 && v[1] > 1e-3 && v[2] > 1e-3)]
        @test !isempty(eq)
        # the ray leaves the tile core where x or y reaches nc
        inA(p) = (v = MR.pix2vec_ring(ns, p); h * count(j -> rmin <= (j - 0.5) * h <= chimax && (j - 0.5) * h * max(v[1], v[2]) < nc, 1:100))
        @test all(p -> mB[p] == 0 && mA[p] ≈ inA(p), eq)
    end

    @testset "gaussian_split ISW potential is continuous across tile faces" begin
        # neighbouring tiles hold the cells next to their shared face (one as core, one as buffer); with the
        # short-range kernel −(1 − G)/k² on the full tile δ they agree to a few % of the short-range potential
        # (−1/k² on the truncated δ_S left per-tile offsets, ~50% of it)
        MR = Huntian.MultiResolution
        pk = Huntian.load_pk(joinpath(datadir, "pk_websky.dat"))
        a = 2.0; nsb = 24; nbf = 12; nm = nsb + 2nbf; Nb = 2nsb; blk = 4; Mc = Nb ÷ blk; Lt = nm * a
        cn = MR._downsample_noise(Nb, Mc, 7)
        gs = MR._gsplit_setup(cn, pk, Mc, Nb * a, MR._splice_compensation(Mc, blk), 4.0, a; need_pot=true)
        t = map(1:2) do it
            rsd = MR._generate_extended_residual(it, 1, 1, nsb, nm, Nb, 7, cn, Mc, 0)
            ds = MR._isolated_convolve_dispatch(false, rsd, pk, Lt, nm, 0, 0, 0)
            _, _, pot = MR._gsplit_tile(gs, false, ds, cn, it, 1, 1, nsb, nm, Nb, Mc, Lt; need_pot=true)
            Float64.(pot), Float64.(MR._interpolate_to_tile(gs.potL, it, 1, 1, nsb, nm, Nb, Mc))
        end
        c = nbf+1:nm-nbf
        d = vcat([vec(t[1][1][g+nbf, c, c] .- t[2][1][g-nsb+nbf, c, c]) for g in (nsb, nsb + 1)]...)
        sS = sqrt(sum(abs2, t[1][1][c, c, c] .- t[1][2][c, c, c]) / length(c)^3)
        @test sqrt(sum(abs2, d) / length(d)) < 0.1 * sS
    end

    @testset "ISW gather vs deposit (same potential)" begin
        mg = run(; kernels=[:isw], subdiv_max=3, isw_mode=:gather)
        md = run(; kernels=[:isw], subdiv_max=3, isw_mode=:deposit)
        cov = [p for p in 1:npix if all(>(0.05), Healpix.pix2vecRing(res, p))]
        @test sum(mg[:isw][cov]) / sum(md[:isw][cov]) ≈ 1 rtol=0.02
        # (no correlation test against :deposit: on this small box the deposit map is dominated by the near-observer
        # shot noise that :gather removes; on a production octant the two agree to 0.97-1.00 where the signal dominates)
        # deposit noise: pixel-to-pixel jumps are larger in the scatter map than in the ray map
        nbr(mp) = sum(p -> (q = Healpix.vec2pixRing(res, (Healpix.pix2vecRing(res, p) .+ (0.0, 0.0, 0.02))...); (mp[p] - mp[q])^2), cov) / length(cov)
        @test nbr(mg[:isw]) < nbr(md[:isw])
    end

    @testset "cross-tile exclusion mask == brute force" begin
        hb = (x=[0.0], y=[-200.0], z=[-200.0], R=[30.0])     # straddles the x=0 tile boundary
        dcore = nsub * alatt; x0 = -(ntile / 2) * dcore
        bt = ntuple(d -> clamp(floor(Int, ((hb.x[1], hb.y[1], hb.z[1])[d] - x0) / dcore) + 1, 1, ntile), 3)
        bins = Dict{NTuple{3,Int},Vector{Int}}(bt => [1])
        nmask = 0
        for kt in 1:ntile, jt in 1:ntile, it in 1:ntile
            xb, yb, zb = Huntian.MultiResolution.tile_center(it, jt, kt, ntile, dcore)
            nmask += count(Huntian.MultiResolution._build_exclusion_mask(
                hb, bins, it, jt, kt, ntile, dcore, nmesh, nbuff, alatt, xb, yb, zb))
        end
        nbrute = count((cellpos(i) - hb.x[1])^2 + (cellpos(j) - hb.y[1])^2 + (cellpos(k) - hb.z[1])^2 <= hb.R[1]^2
                       for k in 1:Nc, j in 1:Nc, i in 1:Nc)
        @test nmask == nbrute
    end
end
