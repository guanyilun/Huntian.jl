# Halo profiles for paint_halos.jl:
#   κ   — truncated NFW (c = 7 to r200m, then an r⁻² tail to xmax = 1.5 r200m, about the splashback radius) × Born kernel;
#         the compensated version subtracts a Δ = 3 uniform sphere of the full painted mass 1.318·M200m (zero net).
#   y   — Battaglia 2012 AGN pressure (pks2map bbps_ptilde model 1), M200c from M200m via NFW + Duffy et al. 2008
#         (mass=nfw, default; mass=sis is the Websky map-maker factor), spherical truncation x ≤ 4.
#   kSZ — Battaglia AGN gas density (bbps_rhotilde), M200c > 1e13 Msun and r200c > 0.5′ cuts; W (plain) and
#         Wc (minus a uniform sphere of the same total τ, radius 4·r200c → zero net electrons).
# Units: catalog lengths Mpc/h, masses from R_TH; see each block.

using Huntian: chi_cmb, CosmologyParams

const hub = 0.68
const Om = 0.31; const OL = 0.69; const OmB = 0.049
const fb = OmB / Om
const rho_mh = 2.775e11 * Om            # Msun/h per (Mpc/h)^3
const C_KMS = 299792.458
const TCMB_UK = 2.7255e6
const H0C = 1.0 / 2997.92458            # H0/c in h/Mpc
const chistar = chi_cmb(CosmologyParams(Om, OmB, OL, hub, 0.965, 0.81))   # CMB source plane (z* = 1089), Mpc/h

Ez2(z) = Om * (1 + z)^3 + OL
function deltacrit(z)                   # Bryan-Norman, crit-density units
    x = Om * (1 + z)^3 / Ez2(z) - 1
    18π^2 + 82x - 39x^2
end
rhocrit_com(z) = 2.775e11 * hub^2 * Ez2(z) / (1 + z)^3    # Msun/Mpc^3 comoving
rvir_com_mpc(mh, z) = cbrt(3 * mh / (4π * 200 * rhocrit_com(z)))   # R200c, Mpc comoving

# Halo mass for the pressure (B12) and gas (B16) profiles, both fitted in M200c. Catalogue masses are abundance-matched
# M200m [Msun/h]. MASS_PROXY (set by the including script):
#   :nfw — NFW with the Duffy et al. (2008) concentration (full sample, Δ = 200ρ̄_m: c = 10.14 (M/2e12)^-0.081 (1+z)^-1.01)
#          (default);
#   :sis — the Websky map-maker factor sqrt(Δ_vir(z)/200), which treats the mass as a virial mass of a singular
#          isothermal sphere; kept only for comparison with Websky.
_μnfw(x) = log1p(x) - x / (1 + x)
_Omz(z) = Om * (1 + z)^3 / Ez2(z)
# M200c/M200m of an NFW halo with concentration c (w.r.t. R200m) when Ω_m(z) = om: solve for y = R200c/R200m in
# μ(c y)/(μ(c) y³) = 1/om by bisection
function _m200c_over_m200m(c, om)
    f(y) = _μnfw(c * y) / (_μnfw(c) * y^3) - 1 / om
    lo, hi = 1e-3, 1.0
    for _ in 1:80
        mid = 0.5 * (lo + hi); f(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    _μnfw(c * lo) / _μnfw(c)
end
const _MR_C = range(1.0, 30.0, length=300); const _MR_Z = range(0.0, 6.0, length=601)
const _MR_TAB = [_m200c_over_m200m(c, _Omz(z)) for c in _MR_C, z in _MR_Z]
function m200c_from_m200m(M, z)                 # M [Msun/h] -> M200c [Msun]
    MASS_PROXY === :sis && return sqrt(deltacrit(z) / 200) * M / hub
    c = clamp(10.14 * (M / 2e12)^-0.081 * (1 + z)^-1.01, first(_MR_C), last(_MR_C))
    tc = (c - first(_MR_C)) / step(_MR_C) + 1; tz = clamp(z / step(_MR_Z) + 1, 1.0, length(_MR_Z) - 1e-9)
    i = min(floor(Int, tc), length(_MR_C) - 1); j = floor(Int, tz); fc = tc - i; fz = tz - j; A = _MR_TAB
    r = (A[i, j] * (1 - fc) + A[i+1, j] * fc) * (1 - fz) + (A[i, j+1] * (1 - fc) + A[i+1, j+1] * fc) * fz
    r * M / hub
end

# ---- κ ----
const CNFW = 7.0; const XMAX = 1.5; const DCOMP = 3.0
f_nfw(c) = log(1 + c) - c / (1 + c)
const MTOT_FAC = 1 + CNFW^2 * (XMAX - 1) / ((1 + CNFW)^2 * f_nfw(CNFW))   # 1.318
_rho_dimless(u) = u < CNFW ? 1 / (u * (1 + u)^2) :
                  (u < XMAX * CNFW ? CNFW / ((1 + CNFW)^2 * u^2) : 0.0)
function _g_of_x(x)
    umax = XMAX * CNFW
    x >= umax && return 0.0
    lmax = sqrt(umax^2 - x^2); n = 512; h = lmax / n; acc = 0.0
    for i in 0:n-1
        l0 = i * h; lm = l0 + h / 2; l1 = l0 + h
        acc += h / 6 * (_rho_dimless(sqrt(l0^2 + x^2)) + 4 * _rho_dimless(sqrt(lm^2 + x^2)) +
                        _rho_dimless(sqrt(l1^2 + x^2)))
    end
    2acc
end
const GX_N = 2048
const GX_LOGX = range(log(1e-8), log(XMAX * CNFW), length=GX_N)
const GX_TAB = [_g_of_x(exp(lx)) for lx in GX_LOGX]
function g_of_x(x)
    lx = log(max(x, 1e-8))
    lx >= GX_LOGX[end] && return 0.0
    t = (lx - GX_LOGX[1]) / step(GX_LOGX) + 1
    i = clamp(floor(Int, t), 1, GX_N - 1)
    GX_TAB[i] * (1 - (t - i)) + GX_TAB[i+1] * (t - i)
end
const RHOS_OVER_RHOM = 200 * CNFW^3 / (3 * f_nfw(CNFW))
kernel_kappa(z, χ) = 1.5 * Om * H0C^2 * (1 + z) * χ * (1 - χ / chistar)
r200m(M) = cbrt(3 * M / (800π * rho_mh))                     # Mpc/h, M in Msun/h
rcomp_kappa(M) = cbrt(3 * MTOT_FAC * M / (4π * DCOMP * rho_mh))
# (plain, compensated) κ of one halo at transverse comoving distance b [Mpc/h]
@inline function kappa_pair(b, M, W)
    rs = r200m(M) / CNFW
    Σ = b / rs < XMAX * CNFW ? RHOS_OVER_RHOM * rs * g_of_x(b / rs) : 0.0
    Rc = rcomp_kappa(M)
    Σc = b < Rc ? Σ - DCOMP * 2 * sqrt(Rc^2 - b^2) : Σ
    (W * Σ, W * Σc)
end
kappa_integral(M, W, χ) = W * MTOT_FAC * M / (rho_mh * χ^2)   # ∫κ_plain dΩ
paint_radius_kappa(M) = max(XMAX * r200m(M), rcomp_kappa(M))  # Mpc/h (comp sphere 4.45·r200m)

# ======================== Battaglia tables (shared machinery) ==============================
# Σ̃(x_b; mh, z): projected profile in rvir units with spherical cut x ≤ 4, and
# ITOT(mh, z) = ∫Σ̃ 2πx dx (+ inner disk), trilinear in (log x_b, log mh, log(1+z)).
struct BattTable
    xlog::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int}
    mlog::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int}
    zlog::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int}
    sig::Array{Float64,3}
    itot::Array{Float64,2}
end
function BattTable(prof, mlo, mhi, nm)
    xlog = range(log(1e-3), log(4.0), length=160)
    mlog = range(log(mlo), log(mhi), length=nm)
    zlog = range(log(1.0), log(5.8), length=48)
    sig = Array{Float64,3}(undef, length(xlog), nm, length(zlog))
    itot = Array{Float64,2}(undef, nm, length(zlog))
    Threads.@threads for k in eachindex(zlog)
        zp1 = exp(zlog[k])
        for j in 1:nm
            m14 = exp(mlog[j]) / 1e14
            for i in eachindex(xlog)
                b = exp(xlog[i])
                s0 = sqrt(max(16.0 - b^2, 0.0))
                n = max(Int(ceil(s0 / 1e-2)), 1); ds = s0 / n
                acc = 0.0
                for is in 1:n
                    s = (is - 0.5) * ds
                    acc += prof(sqrt(s^2 + b^2), m14, zp1)
                end
                sig[i, j, k] = 2 * acc * ds
            end
            it = 0.0
            for i in 1:length(xlog)-1
                b0 = exp(xlog[i]); b1 = exp(xlog[i+1])
                it += 0.5 * (sig[i, j, k] * b0 + sig[i+1, j, k] * b1) * (b1 - b0) * 2π
            end
            itot[j, k] = it + sig[1, j, k] * π * exp(xlog[1])^2
        end
    end
    BattTable(xlog, mlog, zlog, sig, itot)
end
@inline function sigma_interp(T::BattTable, xb, mh, z)
    xb >= 4.0 && return 0.0
    tx = clamp((log(max(xb, 1.1e-3)) - T.xlog[1]) / step(T.xlog) + 1, 1.0, length(T.xlog) - 1e-6)
    ty = clamp((log(mh) - T.mlog[1]) / step(T.mlog) + 1, 1.0, length(T.mlog) - 1e-6)
    tz = clamp((log(1 + z) - T.zlog[1]) / step(T.zlog) + 1, 1.0, length(T.zlog) - 1e-6)
    i = floor(Int, tx); j = floor(Int, ty); k = floor(Int, tz)
    fx = tx - i; fy = ty - j; fz = tz - k; A = T.sig
    c00 = A[i, j, k] * (1 - fx) + A[i+1, j, k] * fx
    c10 = A[i, j+1, k] * (1 - fx) + A[i+1, j+1, k] * fx
    c01 = A[i, j, k+1] * (1 - fx) + A[i+1, j, k+1] * fx
    c11 = A[i, j+1, k+1] * (1 - fx) + A[i+1, j+1, k+1] * fx
    (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz
end
@inline function itot_interp(T::BattTable, mh, z)
    ty = clamp((log(mh) - T.mlog[1]) / step(T.mlog) + 1, 1.0, length(T.mlog) - 1e-6)
    tz = clamp((log(1 + z) - T.zlog[1]) / step(T.zlog) + 1, 1.0, length(T.zlog) - 1e-6)
    j = floor(Int, ty); k = floor(Int, tz); fy = ty - j; fz = tz - k; A = T.itot
    (A[j, k] * (1 - fy) + A[j+1, k] * fy) * (1 - fz) + (A[j, k+1] * (1 - fy) + A[j+1, k+1] * fy) * fz
end

# ---- tSZ ----
const SIGMAT = 6.65246e-25
const ME_G = 9.109383e-28
const C_CMS = 2.99792458e10
const MSUN_G = 1.9891e33
const H100_S = 1e2 / 3.08567758e19
const YPERMASS = 3 * SIGMAT * H100_S^2 / (8π * ME_G * C_CMS^2) * MSUN_G / 1.932
const Y0 = fb * 200 * hub^2 * YPERMASS / 2      # per Msun (maptable.f90)
@inline function ptilde(x, m14, zp1)
    P0 = 18.100 * m14^0.154 * zp1^-0.758
    xc = 0.497 * m14^-0.00865 * zp1^0.731
    β = 4.35 * m14^0.0393 * zp1^0.415
    P0 * (x / xc)^-0.3 * (1 + (x / xc))^-β
end
y_amp(mh, z) = Y0 * mh * Ez2(z)                 # y = y_amp · Σ̃P(θ/θv)

# ---- kSZ: Battaglia et al. (2016, B16) gas density, AGN-feedback fit (their eq. A1 and Table 2) ----
# ρ_gas(r) = f_b ρ_crit(z) ρ̄_fit(x), x = r/R200c, ρ̄_fit = ρ0 (x/xc)^γ [1 + (x/xc)^α]^(−(β−γ)/α), xc = 0.5, γ = −0.2
# (f_b as in their Fig. 5 normalization). τ = σ_T/(μ_e m_p) ∫ ρ_gas dl. GAS_PROFILE, set by the including script:
#   :b16    — eq. A1 as published (default);
#   :websky — the Websky pks2map form, kept only for comparison with Websky maps: outer exponent −β, and the
#             τ amplitude's length unit is the 200ρ̄_m radius of M200c instead of R200c, which raises τ by (ρ_c/ρ̄_m)^(1/3).
const MUE = 1.136
const MP_G = 1.67e-24
const RHO100 = 1.88e-29                         # ρ_crit for h = 1 [g cm⁻³]
const MPC_CM = 3.08567758e24
const TAU0PERSIGMA = SIGMAT / (MUE * MP_G) * (3 * MSUN_G / (4π))^(1 / 3) * RHO100^(2 / 3)
const MMIN_KSZ = 1e13                           # Msun, M200c proxy
const THMIN_KSZ = 1.4544e-4                     # 0.5′ in rad, r200c angular radius
# ρ̄_fit of eq. A1, returned in units of ρ̄_m (× ρ_c/ρ̄_m) as the projected table and tau_amp expect
@inline function rhotilde(x, m14, zp1)
    rhofac = 1 + OL / Om / zp1^3
    P0 = 4.0e3 * m14^0.29 * zp1^-0.66
    αρ = 0.88 * m14^-0.03 * zp1^0.19
    β = 3.83 * m14^0.04 * zp1^-0.025
    expo = GAS_PROFILE === :b16 ? -(β + 0.2) / αρ : -β       # −(β−γ)/α with γ = −0.2
    P0 * (x / 0.5)^-0.2 * (1 + (x / 0.5)^αρ)^expo * rhofac
end
# τ = tau_amp · Σ̃(θ/θv): ρ̄_m(z) times the length unit of the projection (R200c for :b16), times f_b
tau_amp(mh, z) = TAU0PERSIGMA * (Om * hub^2)^(2 / 3) * (1 + z)^2 * (mh / 200)^(1 / 3) * fb *
                 (GAS_PROFILE === :b16 ? cbrt(Om * (1 + z)^3 / Ez2(z)) : 1.0)

# ============================== self-tests (amplitude anchors) =============================
function profile_selftests(TY::BattTable, TK::BattTable)
    # κ: ∫κ dΩ plain == analytic, compensated == 0 (independent quadrature)
    let M = 3e14, z = 0.7, χ = 1700.0
        W = kernel_kappa(z, χ); θmax = paint_radius_kappa(M) / χ
        n = 40000; h = θmax / n; a0 = 0.0; ac = 0.0
        for i in 1:n
            θ = (i - 0.5) * h; k0, kc = kappa_pair(θ * χ, M, W)
            a0 += k0 * 2π * θ * h; ac += kc * 2π * θ * h
        end
        want = kappa_integral(M, W, χ)
        abs(a0 / want - 1) < 0.01 || error("κ plain integral self-test failed: $(a0/want)")
        abs(ac) < 0.01 * want || error("κ compensated zero-net self-test failed: $(ac/want)")
    end
    # y and τ central amplitudes
    y15 = y_amp(1e15, 0.2) * sigma_interp(TY, 1.2e-3, 1e15, 0.2)
    (2e-5 < y15 < 5e-4) || error("y amplitude anchor failed: $y15")
    τ0 = tau_amp(3e14, 0.55) * sigma_interp(TK, 1.2e-3, 3e14, 0.55)
    (5e-4 < τ0 < 1.2e-2) || error("central-τ anchor failed: $τ0")
    if GAS_PROFILE === :b16
        # painted τ against σ_T ∫ n_e dl evaluated directly from eq. A1 in physical units
        let M = 3e14, z = 0.5, b = 0.3
            zp1 = 1 + z; m14 = M / 1e14
            ρc = RHO100 * hub^2 * Ez2(z)                             # g cm⁻³, physical
            R = rvir_com_mpc(M, z) / zp1 * MPC_CM                    # R200c, physical cm
            fit(x) = rhotilde(x, m14, zp1) / (1 + OL / Om / zp1^3)   # ρ̄_fit
            n = 200000; s0 = sqrt(16 - b^2); ds = s0 / n
            I = 2ds * sum(fit(sqrt(((i - 0.5) * ds)^2 + b^2)) for i in 1:n)
            τd = SIGMAT / (MUE * MP_G) * fb * ρc * R * I
            τp = tau_amp(M, z) * sigma_interp(TK, b, M, z)
            abs(τp / τd - 1) < 0.01 || error("kSZ τ normalization self-test failed: painted/direct = $(τp/τd)")
            # gas mass inside R200c relative to f_b M200c
            nx = 100000; dx = 1 / nx
            fgas = 3 / 200 * dx * sum(((i - 0.5) * dx)^2 * fit((i - 0.5) * dx) for i in 1:nx)
            (0.3 < fgas < 1.2) || error("gas-fraction self-test failed: M_gas(<R200c)/(f_b M200c) = $fgas")
            @printf("B16 gas self-tests passed: painted/direct τ = %.4f, M_gas(<R200c)/(f_b M200c) = %.3f\n", τp / τd, fgas)
        end
    end
    @printf("profile self-tests passed (gas profile %s): y(1e15,z=.2,b≈0)=%.2e  τ(3e14,z=.55,b≈0)=%.2e\n",
            GAS_PROFILE, y15, τ0)
end
