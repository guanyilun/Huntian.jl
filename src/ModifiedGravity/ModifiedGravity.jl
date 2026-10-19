module ModifiedGravity

# Linear growth in modified gravity on a ΛCDM background (quasi-static, sub-horizon):
#   D'' + (2 + dlnH/dlna) D' − (3/2) Ω_m(a) μ(k,a) D = 0,   ' = d/dlna.
# The growing mode is matched to GR at early times, so the GR-normalised linear field δ_GR(k, z=0) maps to
# δ_MG(k, a) = R_D(k,a) D_GR(a) δ_GR, and the 1LPT velocity aHfDψ to R_v(k,a) aH f_GR D_GR ψ.

using FFTW
import ..Cosmology: CosmologyParams

export GravityModel, GR, HuSawicki, NDGP, Yukawa, mu, compton_k,
    MGGrowthTable, mg_growth_table, growth_ratio, velocity_ratio, mg_patch_corrections,
    second_order_growth_ratio

const C_KMS_H = 2997.92458   # c/H0 in Mpc/h

abstract type GravityModel end

"""General relativity: μ = 1."""
struct GR <: GravityModel end

"""
    HuSawicki(; fR0, n=1)

Hu & Sawicki (2007) f(R) with |f_R0| = `fR0`: μ = 1 + (1/3) k²/(k² + a²m²(a)), with the scalaron mass
m(a) = (H0/c) √[(Ω_m + 4Ω_Λ)/((n+1)|f_R0|)] · [(Ω_m a⁻³ + 4Ω_Λ)/(Ω_m + 4Ω_Λ)]^((n+2)/2).
"""
Base.@kwdef struct HuSawicki <: GravityModel
    fR0::Float64
    n::Float64 = 1.0
end

"""
    NDGP(; H0rc)

Normal-branch DGP with crossover scale r_c: μ = 1 + 1/(3β), β(a) = 1 + 2H r_c (1 + Ḣ/3H²), scale independent.
"""
Base.@kwdef struct NDGP <: GravityModel
    H0rc::Float64
end

"""
    Yukawa(; eps, lambda)

Phenomenological fifth force of comoving range `lambda` [Mpc/h]: μ = 1 + ε(a) k²/(k² + λ⁻²), with
ε(a) = eps · Ω_Λ(a)/Ω_Λ so that the force switches on with the dark energy.
"""
Base.@kwdef struct Yukawa <: GravityModel
    eps::Float64
    lambda::Float64
end

_Em2(a, c) = c.Om / a^3 + c.OL                     # E²(a), flat matter + Λ
_Omega_m(a, c) = (c.Om / a^3) / _Em2(a, c)

"""Comoving Compton wavenumber a·m(a) [h/Mpc] of the f(R) scalaron."""
function compton_k(m::HuSawicki, a, c::CosmologyParams)
    s = c.Om + 4c.OL
    return a / C_KMS_H * sqrt(s / ((m.n + 1) * m.fR0)) * ((c.Om / a^3 + 4c.OL) / s)^((m.n + 2) / 2)
end

"""    mu(model, k, a, cosmo)

Effective gravitational coupling G_eff/G for the matter density at comoving wavenumber `k` [h/Mpc]."""
mu(::GR, k, a, c::CosmologyParams) = 1.0
function mu(m::HuSawicki, k, a, c::CosmologyParams)
    km2 = compton_k(m, a, c)^2
    return 1.0 + k^2 / (3 * (k^2 + km2))
end
function mu(m::NDGP, k, a, c::CosmologyParams)
    beta = 1.0 + 2.0 * m.H0rc * sqrt(_Em2(a, c)) * (1.0 - _Omega_m(a, c) / 2)
    return 1.0 + 1.0 / (3beta)
end
function mu(m::Yukawa, k, a, c::CosmologyParams)
    eps_a = m.eps * (c.OL / _Em2(a, c)) / c.OL
    return 1.0 + eps_a * k^2 / (k^2 + m.lambda^-2)
end

# RK4 in ln a for y = (D, D'). Returns D and D' at each of `as` (ascending).
function _solve_growth(mufn, as::AbstractVector, c::CosmologyParams; ainit=1e-3, nstep_per_efold=200)
    om = _Omega_m(ainit, c)
    p = (-1 + sqrt(1 + 24 * mufn(ainit) * om)) / 4    # growing mode, matter domination
    y = (ainit, p * ainit)
    rhs(lna, y) = begin
        a = exp(lna); oma = _Omega_m(a, c)
        (y[2], -(2 - 1.5oma) * y[2] + 1.5oma * mufn(a) * y[1])
    end
    D = similar(as, Float64); dD = similar(as, Float64)
    lna = log(ainit)
    for (i, at) in enumerate(as)
        lnt = log(at)
        nstep = max(1, ceil(Int, (lnt - lna) * nstep_per_efold))
        h = (lnt - lna) / nstep
        for _ in 1:nstep
            k1 = rhs(lna, y)
            k2 = rhs(lna + h / 2, y .+ (h / 2) .* k1)
            k3 = rhs(lna + h / 2, y .+ (h / 2) .* k2)
            k4 = rhs(lna + h, y .+ h .* k3)
            y = y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            lna += h
        end
        D[i], dD[i] = y
    end
    return D, dD
end

"""
    MGGrowthTable

R_D(k,a) = D_MG/D_GR (displacements) and R_v(k,a) = (fD)_MG/(fD)_GR (velocities) on a grid in (ln k, a).
"""
struct MGGrowthTable{M<:GravityModel}
    model::M
    lnk::Vector{Float64}
    a::Vector{Float64}
    RD::Matrix{Float64}   # (nk, na)
    Rv::Matrix{Float64}
end

"""
    mg_growth_table(model, cosmo; kmin=1e-4, kmax=1e2, nk=161, amin=0.1, na=91)

Tabulate the MG/GR growth and velocity ratios. Both are solved with the same integrator, so the ratio
carries no integration error from the GR normalisation.
"""
function mg_growth_table(model::GravityModel, c::CosmologyParams;
                         kmin=1e-4, kmax=1e2, nk::Int=161, amin=0.1, na::Int=91)
    lnk = collect(range(log(kmin), log(kmax); length=nk))
    as = collect(range(amin, 1.0; length=na))
    Dg, dDg = _solve_growth(a -> 1.0, as, c)
    RD = Matrix{Float64}(undef, nk, na); Rv = similar(RD)
    Threads.@threads for i in 1:nk
        k = exp(lnk[i])
        D, dD = _solve_growth(a -> mu(model, k, a, c), as, c)
        @views RD[i, :] .= D ./ Dg
        @views Rv[i, :] .= dD ./ dDg
    end
    return MGGrowthTable(model, lnk, as, RD, Rv)
end

function _bilinear(t::MGGrowthTable, M::Matrix{Float64}, k, a)
    k <= 0 && return 1.0     # k = 0 mode: no modification
    x = clamp(log(k), t.lnk[1], t.lnk[end]); y = clamp(a, t.a[1], t.a[end])
    dx = t.lnk[2] - t.lnk[1]; dy = t.a[2] - t.a[1]
    i = min(floor(Int, (x - t.lnk[1]) / dx) + 1, length(t.lnk) - 1)
    j = min(floor(Int, (y - t.a[1]) / dy) + 1, length(t.a) - 1)
    u = (x - t.lnk[i]) / dx; v = (y - t.a[j]) / dy
    return (1 - u) * (1 - v) * M[i, j] + u * (1 - v) * M[i+1, j] + (1 - u) * v * M[i, j+1] + u * v * M[i+1, j+1]
end

"""    growth_ratio(table, k, a) = D_MG(k,a)/D_GR(a); clamped to the table edges."""
growth_ratio(t::MGGrowthTable, k, a) = _bilinear(t, t.RD, k, a)

"""    velocity_ratio(table, k, a) = (fD)_MG(k,a)/(fD)_GR(a); clamped to the table edges."""
velocity_ratio(t::MGGrowthTable, k, a) = _bilinear(t, t.Rv, k, a)

_tophat(x) = x < 1e-3 ? 1.0 - x^2 / 10 : 3 * (sin(x) - x * cos(x)) / x^3

# Periodic trilinear interpolation; u = 0-based fractional cell index.
function _trilinear(f::AbstractArray{<:Real,3}, u1, u2, u3)
    n1, n2, n3 = size(f)
    i = floor(Int, u1); j = floor(Int, u2); k = floor(Int, u3)
    a = u1 - i; b = u2 - j; c = u3 - k
    i0 = mod(i, n1) + 1; i1 = mod(i + 1, n1) + 1
    j0 = mod(j, n2) + 1; j1 = mod(j + 1, n2) + 1
    k0 = mod(k, n3) + 1; k1 = mod(k + 1, n3) + 1
    @inbounds return (1 - a) * (1 - b) * (1 - c) * f[i0, j0, k0] + a * (1 - b) * (1 - c) * f[i1, j0, k0] +
                     (1 - a) * b * (1 - c) * f[i0, j1, k0] + a * b * (1 - c) * f[i1, j1, k0] +
                     (1 - a) * (1 - b) * c * f[i0, j0, k1] + a * (1 - b) * c * f[i1, j0, k1] +
                     (1 - a) * b * c * f[i0, j1, k1] + a * b * c * f[i1, j1, k1]
end

# Second-order growth for constant-k μ: D₂'' + (2 + dlnH/dlna) D₂' − (3/2)Ω_m μ D₂ = −(3/2)Ω_m μ D², started
# on the matter-era solution D₂ = −(3/7) D². Used to check the source-product approximation of the MG 2LPT terms.
function _solve_growth2(mufn, as::AbstractVector, c::CosmologyParams; ainit=1e-3, nstep_per_efold=200)
    om = _Omega_m(ainit, c)
    p = (-1 + sqrt(1 + 24 * mufn(ainit) * om)) / 4
    y = (ainit, p * ainit, -3 / 7 * ainit^2, -3 / 7 * 2p * ainit^2)
    rhs(lna, y) = begin
        a = exp(lna); oma = _Omega_m(a, c); m = mufn(a)
        (y[2], -(2 - 1.5oma) * y[2] + 1.5oma * m * y[1],
         y[4], -(2 - 1.5oma) * y[4] + 1.5oma * m * (y[3] - y[1]^2))
    end
    D2 = similar(as, Float64)
    lna = log(ainit)
    for (i, at) in enumerate(as)
        lnt = log(at)
        nstep = max(1, ceil(Int, (lnt - lna) * nstep_per_efold))
        h = (lnt - lna) / nstep
        for _ in 1:nstep
            k1 = rhs(lna, y)
            k2 = rhs(lna + h / 2, y .+ (h / 2) .* k1)
            k3 = rhs(lna + h / 2, y .+ (h / 2) .* k2)
            k4 = rhs(lna + h, y .+ h .* k3)
            y = y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            lna += h
        end
        D2[i] = y[3]
    end
    return D2
end

"""
    second_order_growth_ratio(model, k, a, cosmo) -> (D₂_MG/D₂_GR, R_D²)

The ratio of second-order growth factors for a mode whose first- and second-order parts both sit at `k`
(μ evaluated at `k`), next to R_D², the ratio implied by building the 2LPT source from MG first-order fields
while keeping the GR second-order response. Their difference is what `mg_patch_corrections` neglects.
"""
function second_order_growth_ratio(model::GravityModel, k, a, c::CosmologyParams)
    D2m = _solve_growth2(x -> mu(model, k, x, c), [a], c)[1]
    D2g = _solve_growth2(x -> 1.0, [a], c)[1]
    Dm, _ = _solve_growth(x -> mu(model, k, x, c), [a], c)
    Dg, _ = _solve_growth(x -> 1.0, [a], c)
    return D2m / D2g, (Dm[1] / Dg[1])^2
end

# Wavenumber grids of an rfft of an n³ box of side L.
_kgrids(n, L) = (FFTW.rfftfreq(n, 2π * n / L), FFTW.fftfreq(n, 2π * n / L))

# rfft of the symmetric bilinear 2LPT source s(φᴬ, φᴮ) = Σ_{i<j} ½(φᴬᵢᵢφᴮⱼⱼ + φᴮᵢᵢφᴬⱼⱼ) − φᴬᵢⱼφᴮᵢⱼ, with
# φˣᵢⱼ(k) = −kᵢkⱼ/k² wX(k) δ(k); s(φ, φ) is Huntian's source (sum of the principal 2×2 minors). The Nyquist
# planes of φ are zeroed: kᵢkⱼ is not Hermitian there, and irfft of such input depends on the FFTW plan.
function _src2_pair_k(delta_k, n, L, wA, wB)
    kx, ky = _kgrids(n, L)
    kv(ix, iy, iz, d) = d == 1 ? kx[ix] : (d == 2 ? ky[iy] : ky[iz])
    work = similar(delta_k)
    irplan = FFTW.plan_irfft(work, n)
    phi(w, di, dj) = begin
        Threads.@threads for iz in 1:n
            @inbounds for iy in 1:n, ix in 1:length(kx)
                k2 = kx[ix]^2 + ky[iy]^2 + ky[iz]^2
                nyq = ix == length(kx) || iy == n ÷ 2 + 1 || iz == n ÷ 2 + 1
                work[ix, iy, iz] = k2 == 0.0 || nyq ? zero(eltype(work)) :
                    -kv(ix, iy, iz, di) * kv(ix, iy, iz, dj) / k2 * w(sqrt(k2)) * delta_k[ix, iy, iz]
            end
        end
        irplan * work
    end
    src = zeros(n, n, n)
    for (di, dj) in ((1, 2), (1, 3), (2, 3))
        A = phi(wA, di, dj)
        src .-= wA === wB ? A .^ 2 : A .* phi(wB, di, dj)
    end
    A = [phi(wA, d, d) for d in 1:3]
    B = wA === wB ? A : [phi(wB, d, d) for d in 1:3]
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        src .+= 0.5 .* (A[i] .* B[j] .+ B[i] .* A[j])
    end
    return FFTW.rfft(src)
end

# Top-hat patch means at the halos of the displacement ψ_d(k) = sgn·i k_d/k² w(k) S(k): 3×N, with the
# top-hat at `Rs` and linear interpolation in ln R to each halo's RTHL.
function _patch_mean(S_k, w, sgn, n, L, u, RTHL, Rs)
    kx, ky = _kgrids(n, L)
    N = length(RTHL)
    work = similar(S_k)
    irplan = FFTW.plan_irfft(work, n)
    samples = zeros(3, N, length(Rs))
    for (b, R) in enumerate(Rs), d in 1:3
        Threads.@threads for iz in 1:n
            @inbounds for iy in 1:n, ix in 1:length(kx)
                k2 = kx[ix]^2 + ky[iy]^2 + ky[iz]^2
                if k2 == 0.0 || ix == length(kx) || iy == n ÷ 2 + 1 || iz == n ÷ 2 + 1
                    work[ix, iy, iz] = 0       # Nyquist planes zeroed, as in the LPT displacements
                    continue
                end
                k = sqrt(k2); kd = (kx[ix], ky[iy], ky[iz])[d]
                work[ix, iy, iz] = sgn * im * kd / k2 * w(k) * _tophat(k * R) * S_k[ix, iy, iz]
            end
        end
        f = irplan * work
        Threads.@threads for h in 1:N
            samples[d, h, b] = _trilinear(f, u[1, h], u[2, h], u[3, h])
        end
    end
    length(Rs) == 1 && return samples[:, :, 1]
    out = zeros(3, N)
    lnRs = log.(Rs)
    for h in 1:N
        s = (log(RTHL[h]) - lnRs[1]) / (lnRs[2] - lnRs[1])
        i = clamp(floor(Int, s) + 1, 1, length(Rs) - 1); wt = s - (i - 1)
        out[:, h] .= (1 - wt) .* samples[:, h, i] .+ wt .* samples[:, h, i+1]
    end
    return out
end

"""
    mg_patch_corrections(delta, boxsize, xq, RTHL, a, table; nR=12, D1=1.0, D2=nothing) -> (ddisp, dvel)

MG corrections to halo displacements, to pass to `finalize_eulerian(...; ddisp, dvel)`. Halo finding stays on
the GR linear field; each halo's mean displacement over its Lagrangian patch (a top-hat of radius `RTHL`
about the peak) gains

  1LPT:  D1 ⟨(R_D − 1) ψ⟩ in the position,   D1 ⟨(R_v − 1) ψ⟩ in the velocity;
  2LPT:  D2 ⟨ψ₂[s(φ_D, φ_D) − s(φ, φ)]⟩ in the position,   2 D2 ⟨ψ₂[s(φ_D, φ_v) − s(φ, φ)]⟩ in the velocity,

where φ_X is the potential-Hessian field weighted by R_X(k, a), so the second-order source is built from MG
first-order fields (the velocity source from d(D_{k₁}D_{k₂})/dln a ∝ D·fD, symmetrised). The MG response of the
second-order mode itself is neglected; `second_order_growth_ratio` measures it. `D1` and `D2` are the GR first-
and second-order growth at `a` in Huntian's conventions (D and −(3/7)Ω_m(a)^(−1/143) D²); with `D2 = nothing`
the 2LPT terms are left at GR.

`delta` is the linear GR density (z = 0 normalisation) on a periodic n³ grid of side `boxsize` [Mpc/h] in
Huntian's centred frame (cell I at x = Δ(I − (n+1)/2)); `xq` is 3×N Lagrangian peak positions [Mpc/h].
Top-hat radii are tabulated at `nR` log-spaced values spanning `RTHL` and interpolated in ln R.
"""
function mg_patch_corrections(delta::AbstractArray{<:Real,3}, boxsize::Real, xq::AbstractMatrix,
                              RTHL::AbstractVector, a::Real, t::MGGrowthTable; nR::Int=12,
                              D1::Real=1.0, D2::Union{Nothing,Real}=nothing)
    n = size(delta, 1)
    @assert size(delta) == (n, n, n) "delta must be cubic"
    @assert size(xq) == (3, length(RTHL))
    N = length(RTHL)
    (N == 0 || t.model isa GR) && return zeros(3, N), zeros(3, N)

    u = (xq ./ (boxsize / n)) .+ (n - 1) / 2       # 0-based fractional index
    Rmin, Rmax = extrema(RTHL)
    Rs = Rmax > Rmin ? exp.(range(log(Rmin), log(Rmax); length=nR)) : [Rmin]
    RD(k) = _bilinear(t, t.RD, k, a); Rv(k) = _bilinear(t, t.Rv, k, a)

    delta_k = FFTW.rfft(Float64.(delta))
    ddisp = D1 .* _patch_mean(delta_k, k -> RD(k) - 1, 1, n, boxsize, u, RTHL, Rs)
    dvel  = D1 .* _patch_mean(delta_k, k -> Rv(k) - 1, 1, n, boxsize, u, RTHL, Rs)
    if D2 !== nothing
        one_(k) = 1.0
        s0 = _src2_pair_k(delta_k, n, boxsize, one_, one_)
        sD = _src2_pair_k(delta_k, n, boxsize, RD, RD) .- s0
        sv = _src2_pair_k(delta_k, n, boxsize, RD, Rv) .- s0
        ddisp .+= D2 .* _patch_mean(sD, one_, -1, n, boxsize, u, RTHL, Rs)
        dvel  .+= 2D2 .* _patch_mean(sv, one_, -1, n, boxsize, u, RTHL, Rs)
    end
    return ddisp, dvel
end

end # module ModifiedGravity
