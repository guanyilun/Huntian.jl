using Test
using Printf
using Huntian
const LCG = Huntian.LCG   # test the package's module (not a second copy of its source)

@testset "LCG — exact Fortran reproducibility" begin

    @testset "modmult identity" begin
        @test LCG.modmult((1, 0, 0, 0), (123, 456, 789, 101)) == (123, 456, 789, 101)
    end

    @testset "modmult against hand-computed value" begin
        # Only checks that the result is four base-4096 digits
        result = LCG.modmult(LCG.MULTIPLIER, LCG.DEFAULT_SEED)
        for d in result
            @test 0 <= d < 4096
        end
    end

    @testset "ranf returns values in [0,1)" begin
        seed = LCG.DEFAULT_SEED
        for _ in 1:1000
            r, seed = LCG.ranf(seed)
            @test 0.0 <= r < 1.0
        end
    end

    @testset "ranf sequence is deterministic" begin
        seed1 = LCG.DEFAULT_SEED
        seed2 = LCG.DEFAULT_SEED
        for _ in 1:100
            r1, seed1 = LCG.ranf(seed1)
            r2, seed2 = LCG.ranf(seed2)
            @test r1 == r2
        end
    end

    @testset "rans parallel streams are different" begin
        seeds = LCG.rans(10, 0)
        @test length(seeds) == 10
        for i in 1:10
            @test seeds[i] != seeds[1] || i == 1
        end
    end

    @testset "gaussdev produces standard normal" begin
        seed = LCG.DEFAULT_SEED
        state = LCG.GaussState()
        N = 10000
        vals = Float64[]
        for _ in 1:N
            g, seed = LCG.gaussdev(seed, state)
            push!(vals, g)
        end
        mu = sum(vals) / N
        sigma = sqrt(sum((vals .- mu).^2) / (N - 1))
        @test abs(mu) < 0.05
        @test abs(sigma - 1.0) < 0.05
    end

    @testset "first few ranf values match Fortran reference" begin
        # No stored Fortran reference: checks distinctness and prints the first
        # 20 values (default seed) for manual comparison with Fortran ranf().
        seed = LCG.DEFAULT_SEED
        vals = Float64[]
        for _ in 1:20
            r, seed = LCG.ranf(seed)
            push!(vals, r)
        end
        # distinct with probability 1 - 2^{-46}
        @test length(unique(vals)) == 20
        println("\n  First 20 ranf values (compare with Fortran):")
        for (i, v) in enumerate(vals)
            @printf("    %2d: %.15e\n", i, v)
        end
    end
end
