@testset "Filters" begin
    @testset "Gaussian window" begin
        @test gaussian_window(0.0) ≈ 1.0
        for kR in [0.1, 0.5, 1.0, 2.0, 5.0]
            @test gaussian_window(kR) ≈ exp(-kR^2 / 2) rtol=1e-10
        end
        @test gaussian_window(10.0) < 1e-20
    end

    @testset "Gaussian window: Fortran R/2 convention" begin
        # Fortran smooth_field divides kR by 2 before calling wgauss,
        # giving W(kR) = exp(-k²R²/8), i.e. effective σ = R/2.
        for kR in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
            @test gaussian_window_fortran(kR) ≈ exp(-kR^2 / 8) rtol=1e-10
        end
        # σ = R/2 smooths less, so the k-space window is wider
        @test gaussian_window_fortran(1.0) > gaussian_window(1.0)
    end

    @testset "Top-hat window" begin
        # W(0) = 1 by L'Hôpital
        @test tophat_window(0.0) ≈ 1.0

        for kR in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
            expected = 3.0 * (sin(kR) - kR * cos(kR)) / kR^3
            @test tophat_window(kR) ≈ expected rtol=1e-10
        end

        # first zero at kR ≈ 4.493
        @test tophat_window(4.0) > 0
        @test tophat_window(5.0) < 0
    end

    @testset "Filter bank file I/O" begin
        tmpfile = tempname()
        open(tmpfile, "w") do io
            println(io, "3")
            println(io, "1  1.686  2.0")
            println(io, "2  1.686  4.0")
            println(io, "3  1.686  8.0")
        end

        fb = read_filterbank(tmpfile)
        rm(tmpfile)

        @test length(fb) == 3
        @test fb[1] == (1, 1.686, 2.0)
        @test fb[2] == (2, 1.686, 4.0)
        @test fb[3] == (3, 1.686, 8.0)
    end

    @testset "Window function normalization" begin
        # Small-kR limits: Gaussian 1 - (kR)²/2, top-hat 1 - (kR)²/10
        @test gaussian_window(0.01) > 0.999
        @test tophat_window(0.01) > 0.999
    end
end
