@testset "Cosmology" begin
    # Planck 2018 parameters
    c = CosmologyParams(0.2645 + 0.0493, 0.0493, 1.0 - 0.2645 - 0.0493, 0.6735, 0.9649, 0.8111)

    @testset "E²(z)" begin
        @test E2(0.0, c) ≈ 1.0 atol=1e-10  # flat
        @test E2(1.0, c) > 1.0
        z = 2.5
        @test E2(z, c) ≈ c.Om * (1+z)^3 + c.OL
    end

    @testset "H(z)" begin
        # H in km/s/Mpc
        @test H(0.0, c) ≈ 100.0 * 0.6735 rtol=1e-10
        z = 1.5
        @test H(z, c) ≈ 100.0 * c.h * sqrt(E2(z, c)) rtol=1e-10
    end

    @testset "Growth factor D(z)" begin
        @test growth_factor(0.0, c) ≈ 1.0 rtol=1e-10  # normalized to z=0
        # D ~ 1/(1+z) at high z, loosely (Ω_m correction)
        @test growth_factor(10.0, c) ≈ 1.0 / 11.0 rtol=0.3
        @test growth_factor(1.0, c) < growth_factor(0.0, c)
        @test growth_factor(2.0, c) < growth_factor(1.0, c)
        # CPT92 for Planck18: D(0.5) ≈ 0.78, D(1) ≈ 0.61
        @test 0.75 < growth_factor(0.5, c) < 0.82
        @test 0.58 < growth_factor(1.0, c) < 0.65
    end

    @testset "Comoving distance χ(z)" begin
        @test chi(0.0, c) ≈ 0.0 atol=1e-10
        @test chi(1.0, c) > 0.0
        @test chi(2.0, c) > chi(1.0, c)
        # χ(z=1) ≈ 2294 Mpc/h; χ is in Mpc/h, so no extra factor of h
        @test 2250 < chi(1.0, c) < 2350
        # CMB source plane with radiation: CAMB gives 9390.7 Mpc/h for Ωm 0.31, h 0.68 (one massive neutrino);
        # massless neutrinos give 9405. Without radiation χ(1089) is 0.5% larger.
        @test Huntian.chi_cmb(CosmologyParams(0.31, 0.049, 0.69, 0.68, 0.965, 0.81)) ≈ 9405 rtol=2e-3
        @test Huntian.chi_cmb(c) < chi(1089.0, c)
    end

    @testset "Growth rate f(z)" begin
        # f ≈ Ω_m(z)^0.55 → 1 at high z
        f_highz = growth_rate(100.0, c)
        @test abs(f_highz - 1.0) < 0.05
        @test 0.48 < growth_rate(0.0, c) < 0.58
    end

    @testset "Critical overdensity δ_c(z) — Nakamura & Suto 1997" begin
        dc_base = (3.0/20.0) * (12π)^(2/3)
        @test dc_base ≈ 1.6865 rtol=1e-4

        dc0 = delta_c(0.0, c)
        @test 1.67 < dc0 < 1.69

        # δ_c = (3/20)(12π)^{2/3} (1 + 0.0123 log10 Ω_m(z)); Ω_m(0) < 1 → below baseline
        @test dc0 < dc_base

        dc1 = delta_c(1.0, c)
        @test abs(dc1 - dc0) / dc0 < 0.02

        # Ω_m(z) → 1 at high z, so δ_c → baseline
        dc_highz = delta_c(100.0, c)
        @test abs(dc_highz - dc_base) / dc_base < 0.001

        c_test = CosmologyParams(0.314, 0.049, 0.686, 0.67, 0.96, 0.82)
        Omz0 = c_test.Om / (c_test.Om + c_test.OL)
        expected = dc_base * (1.0 + 0.0123 * log10(Omz0))
        @test delta_c(0.0, c_test) ≈ expected rtol=1e-10
    end

    @testset "Chi-to-z inversion" begin
        chi2z = build_chi_to_z(c; z_max=6.0, npts=2000)

        for z_test in [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 5.9]
            chi_val = chi(z_test, c)
            z_recovered = chi_to_z(chi2z, chi_val)
            @test z_recovered ≈ z_test atol=0.005
        end

        @test chi_to_z(chi2z, 0.0) == 0.0

        # beyond the table clamps to z_max
        @test chi_to_z(chi2z, 1e10) == 6.0

        chi_z1 = chi(1.0, c)
        z_pk = peak_redshift(0.0, 0.0, 0.0, chi_z1, 0.0, 0.0, chi2z)
        @test z_pk ≈ 1.0 atol=0.005

        z_pk2 = peak_redshift(100.0, 0.0, 0.0, 100.0 + chi_z1, 0.0, 0.0, chi2z)
        @test z_pk2 ≈ 1.0 atol=0.005
    end

    @testset "Self-consistency: D(z) and f(z)" begin
        # f(z) = -d ln D / d ln(1+z) = -(1+z)/D * dD/dz
        z = 1.0
        dz = 0.001
        D_plus  = growth_factor(z + dz, c)
        D_minus = growth_factor(z - dz, c)
        dDdz = (D_plus - D_minus) / (2 * dz)
        D_z = growth_factor(z, c)
        f_numerical = -(1 + z) / D_z * dDdz
        @test f_numerical ≈ growth_rate(z, c) rtol=0.05
    end

    @testset "Ω_m(a) for the 2LPT coefficient" begin
        om_a = Huntian.Cosmology.omega_m_a
        @test om_a(c.Om, c.OL, 1.0) ≈ c.Om
        # Ω_m(z) = Om(1+z)³/E²(z) → 1 at high z
        for z in (0.5, 1.0, 4.5)
            @test om_a(c.Om, c.OL, 1 / (1 + z)) ≈ c.Om * (1 + z)^3 / E2(z, c) rtol=1e-12
        end
        @test om_a(c.Om, c.OL, 1 / 5.5) > 0.98
    end
end
