module CUDAExt

# CUDA implementations of Huntian's GPU entry points (shell analysis, FFT
# convolutions, peak finding). Loaded automatically on `using CUDA`.

using Huntian
using Huntian: ShellTables
using CUDA

# Int32 constants for kernels (avoid Int64 promotion).
const _I0 = Int32(0)
const _I1 = Int32(1)
const _I5 = Int32(5)
const _I31 = Int32(31)

struct ShellTablesGPU
    offsets_di::CuVector{Int32}
    offsets_dj::CuVector{Int32}
    offsets_dk::CuVector{Int32}
    shell_start::CuVector{Int32}
    shell_count::CuVector{Int32}
    shell_r2::CuVector{Int32}
    nshells::Int
end

"""
    ShellTablesGPU(stab::ShellTables) -> ShellTablesGPU

Upload a CPU `ShellTables` to GPU memory (O(rmax^3) Int32 elements, ~6 MB for rmax=50).
"""
function ShellTablesGPU(stab::ShellTables)
    return ShellTablesGPU(
        CuArray{Int32}(stab.offsets_di),
        CuArray{Int32}(stab.offsets_dj),
        CuArray{Int32}(stab.offsets_dk),
        CuArray{Int32}(stab.shell_start),
        CuArray{Int32}(stab.shell_count),
        CuArray{Int32}(stab.shell_r2),
        stab.nshells,
    )
end

# Float32 copy of the 1101-entry SPH kernel table (atab4), uploaded in __init__.

const _AKK_CPU_F32 = Float32[x for x in Huntian.atab4()]
const _AKK_GPU = Ref{CuArray{Float32,1}}()

function __init__()
    _AKK_GPU[] = CuArray(_AKK_CPU_F32)
    return
end

# cuFFT plans memoized by (cube size, device id). Plans depend only on shape,
# and creation is costly. Multi-GPU worker tasks bind a device first (see
# `set_cuda_device!`), so the device id here is the task-local one.

const _CUFFT_PLAN_LOCK = ReentrantLock()
const _CUFFT_FWD_CACHE = Dict{Tuple{Int,Int},Any}()  # (n, devid) -> plan_rfft
const _CUFFT_INV_CACHE = Dict{Tuple{Int,Int},Any}()  # (n, devid) -> plan_irfft

@inline function _cufft_fwd_plan(arr::CuArray{Float32,3})
    n = size(arr, 1)
    devid = CUDA.deviceid(CUDA.device())
    key = (n, devid)
    lock(_CUFFT_PLAN_LOCK) do
        get!(_CUFFT_FWD_CACHE, key) do
            CUDA.CUFFT.plan_rfft(arr)
        end
    end
end

@inline function _cufft_inv_plan(arr::CuArray{ComplexF32,3}, n::Integer)
    n_int = Int(n)
    devid = CUDA.deviceid(CUDA.device())
    key = (n_int, devid)
    lock(_CUFFT_PLAN_LOCK) do
        get!(_CUFFT_INV_CACHE, key) do
            CUDA.CUFFT.plan_irfft(arr, n_int)
        end
    end
end

@inline _cufft_rfft(arr::CuArray{Float32,3})            = _cufft_fwd_plan(arr) * arr
@inline _cufft_irfft(arr::CuArray{ComplexF32,3}, n::Integer) = _cufft_inv_plan(arr, n) * arr

# hRinteg: piecewise polynomial weight used by kernel_strain.

@inline function _hRinteg_f32(u0::Float32, u12::Float32)
    x0 = abs(u0)
    if x0 >= 2.0f0
        return 0.0f0
    end

    one7 = 1.0f0 / 7.0f0
    result = 1.6f0 * u12 - 6.4f0 * one7

    x02 = x0 * x0
    x04 = x02 * x02

    if x0 >= 1.0f0
        h1 = x04 * (2.0f0 - x0 * (2.4f0 - x0 * (1.0f0 - x0 * one7)))
        h2 = x02 * (4.0f0 - x0 * (4.0f0 - x0 * (1.5f0 - x0 * 0.2f0)))
        result = result - h2 * u12 + h1
    else
        result = result - 0.2f0 * (u12 - one7)
        if x0 > 0.0f0
            h1 = x04 * (1.0f0 - x02 * (1.0f0 - 3.0f0 * x0 * one7))
            h2 = x02 * (2.0f0 - x02 * (1.5f0 - 0.6f0 * x0))
            result = result - h2 * u12 + h1
        end
    end
    return result
end

# Device-side kernel_strain: radial integration over shells mlow:mupp of one
# peak's profiles (rad, Gshell (3,ns), Gshellf (3,ns), SRshell (3,3,ns)),
# centered on shell mp. Returns 12 Float32:
#   (Ebar_11, Ebar_12, Ebar_13, Ebar_22, Ebar_23, Ebar_33, grad_xyz..., gradf_xyz...)
# Branches as on CPU: mp > 1 uses hRinteg weights scaled by wRnor/rad[mp];
# mp == 1 uses the akk SPH table scaled by aRnor.

@inline function _kernel_strain_f32(rad, Gshell, Gshellf, SRshell, akk_tab,
                                    mp::Int32, mlow::Int32, mupp::Int32,
                                    wRnor::Float32, aRnor::Float32,
                                    hlatt_1::Float32, hlatt_2::Float32)
    E11 = 0.0f0; E12 = 0.0f0; E13 = 0.0f0
    E22 = 0.0f0; E23 = 0.0f0; E33 = 0.0f0
    gx = 0.0f0; gy = 0.0f0; gz = 0.0f0
    gfx = 0.0f0; gfy = 0.0f0; gfz = 0.0f0

    if mp > _I1
        @inbounds rmp = rad[mp]
        rmp2 = rmp * rmp
        m1 = mlow
        while m1 <= mupp
            @inbounds rm1 = rad[m1]
            u0 = hlatt_1 * (rmp - rm1)
            u02 = u0 * u0
            if u02 < 4.0f0
                u12 = hlatt_2 * (rmp2 + rm1 * rm1)
                denom = (u12 - u02)
                wt = _hRinteg_f32(u0, u12) / (denom * denom)
                @inbounds Gx = Gshell[_I1, m1]
                @inbounds Gy = Gshell[Int32(2), m1]
                @inbounds Gz = Gshell[Int32(3), m1]
                @inbounds Gfx_ = Gshellf[_I1, m1]
                @inbounds Gfy_ = Gshellf[Int32(2), m1]
                @inbounds Gfz_ = Gshellf[Int32(3), m1]
                gx  += wt * Gx;   gy  += wt * Gy;   gz  += wt * Gz
                gfx += wt * Gfx_; gfy += wt * Gfy_; gfz += wt * Gfz_
                # Symmetrised SR: 0.5 * wt * (SR[L,K] + SR[K,L])
                @inbounds SR11 = SRshell[_I1, _I1, m1]
                @inbounds SR22 = SRshell[Int32(2), Int32(2), m1]
                @inbounds SR33 = SRshell[Int32(3), Int32(3), m1]
                @inbounds SR12 = SRshell[_I1, Int32(2), m1]
                @inbounds SR21 = SRshell[Int32(2), _I1, m1]
                @inbounds SR13 = SRshell[_I1, Int32(3), m1]
                @inbounds SR31 = SRshell[Int32(3), _I1, m1]
                @inbounds SR23 = SRshell[Int32(2), Int32(3), m1]
                @inbounds SR32 = SRshell[Int32(3), Int32(2), m1]
                E11 -= wt * SR11
                E22 -= wt * SR22
                E33 -= wt * SR33
                E12 -= 0.5f0 * wt * (SR12 + SR21)
                E13 -= 0.5f0 * wt * (SR13 + SR31)
                E23 -= 0.5f0 * wt * (SR23 + SR32)
            end
            m1 += _I1
        end
        scale = wRnor / rmp
        E11 *= scale; E12 *= scale; E13 *= scale
        E22 *= scale; E23 *= scale; E33 *= scale
        gx *= scale; gy *= scale; gz *= scale
        gfx *= scale; gfy *= scale; gfz *= scale
    else
        m1 = mlow
        while m1 <= mupp
            @inbounds rm1 = rad[m1]
            u0 = hlatt_1 * rm1
            con = 100.0f0 * u0 * u0
            icon = Int32(floor(con))
            diff = con - Float32(icon)
            i2c = icon + _I1
            # akk has length 1101
            if i2c >= Int32(1) && i2c < Int32(1101)
                @inbounds a0 = akk_tab[i2c]
                @inbounds a1 = akk_tab[i2c + _I1]
                aww = a0 + diff * (a1 - a0)
                wt = 0.5f0 * aww / rm1
                @inbounds Gx = Gshell[_I1, m1]
                @inbounds Gy = Gshell[Int32(2), m1]
                @inbounds Gz = Gshell[Int32(3), m1]
                @inbounds Gfx_ = Gshellf[_I1, m1]
                @inbounds Gfy_ = Gshellf[Int32(2), m1]
                @inbounds Gfz_ = Gshellf[Int32(3), m1]
                gx  += wt * Gx;   gy  += wt * Gy;   gz  += wt * Gz
                gfx += wt * Gfx_; gfy += wt * Gfy_; gfz += wt * Gfz_
                @inbounds SR11 = SRshell[_I1, _I1, m1]
                @inbounds SR22 = SRshell[Int32(2), Int32(2), m1]
                @inbounds SR33 = SRshell[Int32(3), Int32(3), m1]
                @inbounds SR12 = SRshell[_I1, Int32(2), m1]
                @inbounds SR21 = SRshell[Int32(2), _I1, m1]
                @inbounds SR13 = SRshell[_I1, Int32(3), m1]
                @inbounds SR31 = SRshell[Int32(3), _I1, m1]
                @inbounds SR23 = SRshell[Int32(2), Int32(3), m1]
                @inbounds SR32 = SRshell[Int32(3), Int32(2), m1]
                E11 -= wt * SR11
                E22 -= wt * SR22
                E33 -= wt * SR33
                E12 -= 0.5f0 * wt * (SR12 + SR21)
                E13 -= 0.5f0 * wt * (SR13 + SR31)
                E23 -= 0.5f0 * wt * (SR23 + SR32)
            end
            m1 += _I1
        end
        E11 *= aRnor; E12 *= aRnor; E13 *= aRnor
        E22 *= aRnor; E23 *= aRnor; E33 *= aRnor
        gx *= aRnor; gy *= aRnor; gz *= aRnor
        gfx *= aRnor; gfy *= aRnor; gfz *= aRnor
    end
    return (E11, E12, E13, E22, E23, E33,
            gx, gy, gz, gfx, gfy, gfz)
end

# Trilinear lookup in a CollapseTable on evenly spaced axes; matches the CPU
# `extrapolate(interpolate(..., BSpline(Linear())), out_val)`: out-of-range
# queries return `out_val`.

@inline function _interp3_trilinear_f32(table, x::Float32, y::Float32, z::Float32,
                                         x1::Float32, y1::Float32, z1::Float32,
                                         dx_inv::Float32, dy_inv::Float32, dz_inv::Float32,
                                         nx::Int32, ny::Int32, nz::Int32,
                                         out_val::Float32)
    fx = (x - x1) * dx_inv
    fy = (y - y1) * dy_inv
    fz = (z - z1) * dz_inv

    if (fx < 0.0f0) | (fy < 0.0f0) | (fz < 0.0f0) |
       (fx > Float32(nx - _I1)) | (fy > Float32(ny - _I1)) | (fz > Float32(nz - _I1))
        return out_val
    end

    # 1-based indices of lower corner; clamp so ix+1 ≤ nx
    ix = Int32(floor(fx)) + _I1
    iy = Int32(floor(fy)) + _I1
    iz = Int32(floor(fz)) + _I1
    ix = ix > (nx - _I1) ? (nx - _I1) : ix
    iy = iy > (ny - _I1) ? (ny - _I1) : iy
    iz = iz > (nz - _I1) ? (nz - _I1) : iz

    tx = fx - Float32(ix - _I1)
    ty = fy - Float32(iy - _I1)
    tz = fz - Float32(iz - _I1)
    ox = 1.0f0 - tx; oy = 1.0f0 - ty; oz = 1.0f0 - tz

    @inbounds v000 = table[ix,       iy,       iz]
    @inbounds v100 = table[ix + _I1, iy,       iz]
    @inbounds v010 = table[ix,       iy + _I1, iz]
    @inbounds v110 = table[ix + _I1, iy + _I1, iz]
    @inbounds v001 = table[ix,       iy,       iz + _I1]
    @inbounds v101 = table[ix + _I1, iy,       iz + _I1]
    @inbounds v011 = table[ix,       iy + _I1, iz + _I1]
    @inbounds v111 = table[ix + _I1, iy + _I1, iz + _I1]

    c00 = v000 * ox + v100 * tx
    c10 = v010 * ox + v110 * tx
    c01 = v001 * ox + v101 * tx
    c11 = v011 * ox + v111 * tx
    c0 = c00 * oy + c10 * ty
    c1 = c01 * oy + c11 * ty
    return c0 * oz + c1 * tz
end

# Eigenvalues of a real symmetric 3x3 matrix (6 unique entries) via the
# trigonometric Cardano solution (Smith 1961, Comm. ACM 4(4)); replaces the
# CPU LAPACK `eigvals(Symmetric(mat))`. Returns ascending (min, mid, max).

@inline function _eig3_symmetric_f32(a11::Float32, a22::Float32, a33::Float32,
                                     a12::Float32, a13::Float32, a23::Float32)
    q = (a11 + a22 + a33) * (1.0f0 / 3.0f0)
    b11 = a11 - q; b22 = a22 - q; b33 = a33 - q
    # p^2 = ||A - qI||_F^2 / 6
    p2 = (b11*b11 + b22*b22 + b33*b33 + 2.0f0*(a12*a12 + a13*a13 + a23*a23)) *
         (1.0f0 / 6.0f0)

    if p2 <= 1.0f-30  # matrix is (nearly) a multiple of I
        return (q, q, q)
    end

    p = sqrt(p2)
    inv_p = 1.0f0 / p
    # B = (A - qI) / p; r = det(B)/2 in [-1, 1]
    c11 = b11 * inv_p; c22 = b22 * inv_p; c33 = b33 * inv_p
    c12 = a12 * inv_p; c13 = a13 * inv_p; c23 = a23 * inv_p

    det_b = c11*(c22*c33 - c23*c23) -
            c12*(c12*c33 - c23*c13) +
            c13*(c12*c23 - c22*c13)
    r = det_b * 0.5f0
    # Clamp: Float32 round-off can push r past ±1
    r = r < -1.0f0 ? -1.0f0 : (r > 1.0f0 ? 1.0f0 : r)

    phi = acos(r) * (1.0f0 / 3.0f0)

    eig_max = q + 2.0f0 * p * cos(phi)
    eig_min = q + 2.0f0 * p * cos(phi + (2.0f0 * Float32(pi) / 3.0f0))
    eig_mid = 3.0f0 * q - eig_min - eig_max  # trace = 3q

    return (eig_min, eig_mid, eig_max)
end

# Warp-level sum reduction via shuffle-down. Assumes warp size 32 (NVIDIA).
@inline function warp_reduce_sum_f32(val::Float32)
    val += CUDA.shfl_down_sync(0xffffffff, val, 16)
    val += CUDA.shfl_down_sync(0xffffffff, val, 8)
    val += CUDA.shfl_down_sync(0xffffffff, val, 4)
    val += CUDA.shfl_down_sync(0xffffffff, val, 2)
    val += CUDA.shfl_down_sync(0xffffffff, val, 1)
    return val
end

@inline function warp_reduce_sum_i32(val::Int32)
    val += CUDA.shfl_down_sync(0xffffffff, val, 16)
    val += CUDA.shfl_down_sync(0xffffffff, val, 8)
    val += CUDA.shfl_down_sync(0xffffffff, val, 4)
    val += CUDA.shfl_down_sync(0xffffffff, val, 2)
    val += CUDA.shfl_down_sync(0xffffffff, val, 1)
    return val
end

# Block-level sum: warp shuffle, then warp 0 reduces the per-warp partials
# from shared memory. Result valid on thread 1 only; callers must
# sync_threads() before reusing `sdata`.
@inline function block_reduce_sum_f32(val::Float32, sdata)
    tid = threadIdx().x
    lane = (tid - _I1) & _I31
    warp = (tid - _I1) >> _I5
    v = warp_reduce_sum_f32(val)
    if lane == _I0
        @inbounds sdata[warp + _I1] = v
    end
    sync_threads()
    nwarps = (blockDim().x + _I31) >> _I5
    v2 = (tid - _I1) < nwarps ? (@inbounds sdata[tid]) : 0.0f0
    if warp == _I0
        v2 = warp_reduce_sum_f32(v2)
    end
    return v2
end

@inline function block_reduce_sum_i32(val::Int32, sdata)
    tid = threadIdx().x
    lane = (tid - _I1) & _I31
    warp = (tid - _I1) >> _I5
    v = warp_reduce_sum_i32(val)
    if lane == _I0
        @inbounds sdata[warp + _I1] = v
    end
    sync_threads()
    nwarps = (blockDim().x + _I31) >> _I5
    v2 = (tid - _I1) < nwarps ? (@inbounds sdata[tid]) : _I0
    if warp == _I0
        v2 = warp_reduce_sum_i32(v2)
    end
    return v2
end

# Block-per-peak shell gathers: one block per peak, threads stride over the
# cells of each shell, block-reduce per shell.

function _shell_fbar_kernel!(Fshell_out, nshell_out,
                             delta, peaks_i, peaks_j, peaks_k,
                             off_di, off_dj, off_dk,
                             shell_start, shell_count,
                             nshells::Int32, n1::Int32, n2::Int32, n3::Int32)
    peak_id = blockIdx().x
    tid = threadIdx().x
    bdim = blockDim().x

    # Shared scratch: 32 Float32 then 32 Int32 slots (up to 1024 threads = 32 warps).
    sdata_f = CuDynamicSharedArray(Float32, 32)
    sdata_i = CuDynamicSharedArray(Int32, 32, 32 * sizeof(Float32))

    @inbounds ci = peaks_i[peak_id]
    @inbounds cj = peaks_j[peak_id]
    @inbounds ck = peaks_k[peak_id]

    s = _I1
    while s <= nshells
        @inbounds s0 = shell_start[s]
        @inbounds nc = shell_count[s]

        local_sum = 0.0f0
        local_nsh = _I0

        c = tid
        while c <= nc
            idx = s0 + c - _I1
            @inbounds di = off_di[idx]
            @inbounds dj = off_dj[idx]
            @inbounds dk = off_dk[idx]

            iv1 = ci + di
            iv2 = cj + dj
            iv3 = ck + dk

            if (_I1 <= iv1) & (iv1 <= n1) & (_I1 <= iv2) & (iv2 <= n2) & (_I1 <= iv3) & (iv3 <= n3)
                @inbounds local_sum += delta[iv1, iv2, iv3]
                local_nsh += _I1
            end
            c += bdim
        end

        total_sum = block_reduce_sum_f32(local_sum, sdata_f)
        sync_threads()
        total_nsh = block_reduce_sum_i32(local_nsh, sdata_i)

        if tid == _I1
            @inbounds Fshell_out[s, peak_id] = total_sum
            @inbounds nshell_out[s, peak_id] = total_nsh
        end
        sync_threads()  # sdata reused by next shell
        s += _I1
    end
    return
end

function Huntian.shell_fbar_gather_gpu(delta_h::AbstractArray{<:Real,3},
                                         peaks_i::AbstractVector{<:Integer},
                                         peaks_j::AbstractVector{<:Integer},
                                         peaks_k::AbstractVector{<:Integer},
                                         stab::ShellTables;
                                         threads::Int=128)
    @assert length(peaks_i) == length(peaks_j) == length(peaks_k)
    npeaks = length(peaks_i)
    nshells = stab.nshells

    delta_d = CuArray{Float32}(delta_h)
    pi_d = CuArray{Int32}(peaks_i)
    pj_d = CuArray{Int32}(peaks_j)
    pk_d = CuArray{Int32}(peaks_k)
    stab_d = ShellTablesGPU(stab)

    Fshell_d = CUDA.zeros(Float32, nshells, npeaks)
    nshell_d = CUDA.zeros(Int32, nshells, npeaks)

    n1, n2, n3 = size(delta_d)
    shmem = 32 * sizeof(Float32) + 32 * sizeof(Int32)

    @cuda threads=threads blocks=npeaks shmem=shmem _shell_fbar_kernel!(
        Fshell_d, nshell_d,
        delta_d, pi_d, pj_d, pk_d,
        stab_d.offsets_di, stab_d.offsets_dj, stab_d.offsets_dk,
        stab_d.shell_start, stab_d.shell_count,
        Int32(nshells), Int32(n1), Int32(n2), Int32(n3),
    )

    return Array(Fshell_d), Array(nshell_d)
end

# Also accumulates per shell:
#   Sshell[L] = Σ eta_L[cell]
#   Gshell[L] = Σ delta[cell] * d_L   (un-normalised; caller divides by rad[s])

function _shell_gather_psi_kernel!(Fshell_out, nshell_out, Sshell_out, Gshell_out,
                                   delta, etax, etay, etaz,
                                   peaks_i, peaks_j, peaks_k,
                                   off_di, off_dj, off_dk,
                                   shell_start, shell_count,
                                   nshells::Int32, n1::Int32, n2::Int32, n3::Int32)
    peak_id = blockIdx().x
    tid = threadIdx().x
    bdim = blockDim().x

    sdata_f = CuDynamicSharedArray(Float32, 32)
    sdata_i = CuDynamicSharedArray(Int32, 32, 32 * sizeof(Float32))

    @inbounds ci = peaks_i[peak_id]
    @inbounds cj = peaks_j[peak_id]
    @inbounds ck = peaks_k[peak_id]

    s = _I1
    while s <= nshells
        @inbounds s0 = shell_start[s]
        @inbounds nc = shell_count[s]

        local_F = 0.0f0
        local_Sx = 0.0f0; local_Sy = 0.0f0; local_Sz = 0.0f0
        local_Gx = 0.0f0; local_Gy = 0.0f0; local_Gz = 0.0f0
        local_n = _I0

        c = tid
        while c <= nc
            idx = s0 + c - _I1
            @inbounds di = off_di[idx]
            @inbounds dj = off_dj[idx]
            @inbounds dk = off_dk[idx]

            iv1 = ci + di
            iv2 = cj + dj
            iv3 = ck + dk

            if (_I1 <= iv1) & (iv1 <= n1) & (_I1 <= iv2) & (iv2 <= n2) & (_I1 <= iv3) & (iv3 <= n3)
                @inbounds d_val = delta[iv1, iv2, iv3]
                @inbounds ex = etax[iv1, iv2, iv3]
                @inbounds ey = etay[iv1, iv2, iv3]
                @inbounds ez = etaz[iv1, iv2, iv3]

                local_F += d_val
                local_Sx += ex
                local_Sy += ey
                local_Sz += ez
                fdi = Float32(di); fdj = Float32(dj); fdk = Float32(dk)
                local_Gx += d_val * fdi
                local_Gy += d_val * fdj
                local_Gz += d_val * fdk
                local_n += _I1
            end
            c += bdim
        end

        # sync between each reuse of sdata
        F_tot  = block_reduce_sum_f32(local_F,  sdata_f); sync_threads()
        Sx_tot = block_reduce_sum_f32(local_Sx, sdata_f); sync_threads()
        Sy_tot = block_reduce_sum_f32(local_Sy, sdata_f); sync_threads()
        Sz_tot = block_reduce_sum_f32(local_Sz, sdata_f); sync_threads()
        Gx_tot = block_reduce_sum_f32(local_Gx, sdata_f); sync_threads()
        Gy_tot = block_reduce_sum_f32(local_Gy, sdata_f); sync_threads()
        Gz_tot = block_reduce_sum_f32(local_Gz, sdata_f); sync_threads()
        n_tot  = block_reduce_sum_i32(local_n,  sdata_i)

        if tid == _I1
            @inbounds Fshell_out[s, peak_id] = F_tot
            @inbounds nshell_out[s, peak_id] = n_tot
            @inbounds Sshell_out[_I1, s, peak_id] = Sx_tot
            @inbounds Sshell_out[Int32(2), s, peak_id] = Sy_tot
            @inbounds Sshell_out[Int32(3), s, peak_id] = Sz_tot
            @inbounds Gshell_out[_I1, s, peak_id] = Gx_tot
            @inbounds Gshell_out[Int32(2), s, peak_id] = Gy_tot
            @inbounds Gshell_out[Int32(3), s, peak_id] = Gz_tot
        end
        sync_threads()
        s += _I1
    end
    return
end

function Huntian.shell_gather_psi_gpu(delta_h::AbstractArray{<:Real,3},
                                        etax_h::AbstractArray{<:Real,3},
                                        etay_h::AbstractArray{<:Real,3},
                                        etaz_h::AbstractArray{<:Real,3},
                                        peaks_i::AbstractVector{<:Integer},
                                        peaks_j::AbstractVector{<:Integer},
                                        peaks_k::AbstractVector{<:Integer},
                                        stab::ShellTables;
                                        threads::Int=128)
    @assert size(delta_h) == size(etax_h) == size(etay_h) == size(etaz_h)
    @assert length(peaks_i) == length(peaks_j) == length(peaks_k)
    npeaks = length(peaks_i)
    nshells = stab.nshells

    delta_d = CuArray{Float32}(delta_h)
    etax_d  = CuArray{Float32}(etax_h)
    etay_d  = CuArray{Float32}(etay_h)
    etaz_d  = CuArray{Float32}(etaz_h)
    pi_d = CuArray{Int32}(peaks_i)
    pj_d = CuArray{Int32}(peaks_j)
    pk_d = CuArray{Int32}(peaks_k)
    stab_d = ShellTablesGPU(stab)

    Fshell_d = CUDA.zeros(Float32, nshells, npeaks)
    nshell_d = CUDA.zeros(Int32, nshells, npeaks)
    Sshell_d = CUDA.zeros(Float32, 3, nshells, npeaks)
    Gshell_d = CUDA.zeros(Float32, 3, nshells, npeaks)

    n1, n2, n3 = size(delta_d)
    shmem = 32 * sizeof(Float32) + 32 * sizeof(Int32)

    @cuda threads=threads blocks=npeaks shmem=shmem _shell_gather_psi_kernel!(
        Fshell_d, nshell_d, Sshell_d, Gshell_d,
        delta_d, etax_d, etay_d, etaz_d,
        pi_d, pj_d, pk_d,
        stab_d.offsets_di, stab_d.offsets_dj, stab_d.offsets_dk,
        stab_d.shell_start, stab_d.shell_count,
        Int32(nshells), Int32(n1), Int32(n2), Int32(n3),
    )

    return Array(Fshell_d), Array(nshell_d), Array(Sshell_d), Array(Gshell_d)
end

# Adds the raw (unsymmetrised) strain tensor SRshell[L,K] = Σ eta_L[cell] * d_K;
# symmetrisation happens downstream.

function _shell_gather_strain_kernel!(Fshell_out, nshell_out,
                                      Sshell_out, Gshell_out, SRshell_out,
                                      delta, etax, etay, etaz,
                                      peaks_i, peaks_j, peaks_k,
                                      off_di, off_dj, off_dk,
                                      shell_start, shell_count,
                                      nshells::Int32, n1::Int32, n2::Int32, n3::Int32)
    peak_id = blockIdx().x
    tid = threadIdx().x
    bdim = blockDim().x

    sdata_f = CuDynamicSharedArray(Float32, 32)
    sdata_i = CuDynamicSharedArray(Int32, 32, 32 * sizeof(Float32))

    @inbounds ci = peaks_i[peak_id]
    @inbounds cj = peaks_j[peak_id]
    @inbounds ck = peaks_k[peak_id]

    s = _I1
    while s <= nshells
        @inbounds s0 = shell_start[s]
        @inbounds nc = shell_count[s]

        local_F = 0.0f0
        local_Sx = 0.0f0; local_Sy = 0.0f0; local_Sz = 0.0f0
        local_Gx = 0.0f0; local_Gy = 0.0f0; local_Gz = 0.0f0
        local_SRxx = 0.0f0; local_SRxy = 0.0f0; local_SRxz = 0.0f0
        local_SRyx = 0.0f0; local_SRyy = 0.0f0; local_SRyz = 0.0f0
        local_SRzx = 0.0f0; local_SRzy = 0.0f0; local_SRzz = 0.0f0
        local_n = _I0

        c = tid
        while c <= nc
            idx = s0 + c - _I1
            @inbounds di = off_di[idx]
            @inbounds dj = off_dj[idx]
            @inbounds dk = off_dk[idx]

            iv1 = ci + di
            iv2 = cj + dj
            iv3 = ck + dk

            if (_I1 <= iv1) & (iv1 <= n1) & (_I1 <= iv2) & (iv2 <= n2) & (_I1 <= iv3) & (iv3 <= n3)
                @inbounds d_val = delta[iv1, iv2, iv3]
                @inbounds ex = etax[iv1, iv2, iv3]
                @inbounds ey = etay[iv1, iv2, iv3]
                @inbounds ez = etaz[iv1, iv2, iv3]

                local_F += d_val
                local_Sx += ex; local_Sy += ey; local_Sz += ez

                fdi = Float32(di); fdj = Float32(dj); fdk = Float32(dk)
                local_Gx += d_val * fdi
                local_Gy += d_val * fdj
                local_Gz += d_val * fdk

                local_SRxx += ex * fdi; local_SRxy += ex * fdj; local_SRxz += ex * fdk
                local_SRyx += ey * fdi; local_SRyy += ey * fdj; local_SRyz += ey * fdk
                local_SRzx += ez * fdi; local_SRzy += ez * fdj; local_SRzz += ez * fdk

                local_n += _I1
            end
            c += bdim
        end

        F_tot  = block_reduce_sum_f32(local_F,  sdata_f); sync_threads()
        Sx_tot = block_reduce_sum_f32(local_Sx, sdata_f); sync_threads()
        Sy_tot = block_reduce_sum_f32(local_Sy, sdata_f); sync_threads()
        Sz_tot = block_reduce_sum_f32(local_Sz, sdata_f); sync_threads()
        Gx_tot = block_reduce_sum_f32(local_Gx, sdata_f); sync_threads()
        Gy_tot = block_reduce_sum_f32(local_Gy, sdata_f); sync_threads()
        Gz_tot = block_reduce_sum_f32(local_Gz, sdata_f); sync_threads()
        SRxx_tot = block_reduce_sum_f32(local_SRxx, sdata_f); sync_threads()
        SRxy_tot = block_reduce_sum_f32(local_SRxy, sdata_f); sync_threads()
        SRxz_tot = block_reduce_sum_f32(local_SRxz, sdata_f); sync_threads()
        SRyx_tot = block_reduce_sum_f32(local_SRyx, sdata_f); sync_threads()
        SRyy_tot = block_reduce_sum_f32(local_SRyy, sdata_f); sync_threads()
        SRyz_tot = block_reduce_sum_f32(local_SRyz, sdata_f); sync_threads()
        SRzx_tot = block_reduce_sum_f32(local_SRzx, sdata_f); sync_threads()
        SRzy_tot = block_reduce_sum_f32(local_SRzy, sdata_f); sync_threads()
        SRzz_tot = block_reduce_sum_f32(local_SRzz, sdata_f); sync_threads()
        n_tot    = block_reduce_sum_i32(local_n,  sdata_i)

        if tid == _I1
            @inbounds Fshell_out[s, peak_id] = F_tot
            @inbounds nshell_out[s, peak_id] = n_tot
            @inbounds Sshell_out[_I1, s, peak_id] = Sx_tot
            @inbounds Sshell_out[Int32(2), s, peak_id] = Sy_tot
            @inbounds Sshell_out[Int32(3), s, peak_id] = Sz_tot
            @inbounds Gshell_out[_I1, s, peak_id] = Gx_tot
            @inbounds Gshell_out[Int32(2), s, peak_id] = Gy_tot
            @inbounds Gshell_out[Int32(3), s, peak_id] = Gz_tot
            @inbounds SRshell_out[_I1, _I1, s, peak_id] = SRxx_tot
            @inbounds SRshell_out[_I1, Int32(2), s, peak_id] = SRxy_tot
            @inbounds SRshell_out[_I1, Int32(3), s, peak_id] = SRxz_tot
            @inbounds SRshell_out[Int32(2), _I1, s, peak_id] = SRyx_tot
            @inbounds SRshell_out[Int32(2), Int32(2), s, peak_id] = SRyy_tot
            @inbounds SRshell_out[Int32(2), Int32(3), s, peak_id] = SRyz_tot
            @inbounds SRshell_out[Int32(3), _I1, s, peak_id] = SRzx_tot
            @inbounds SRshell_out[Int32(3), Int32(2), s, peak_id] = SRzy_tot
            @inbounds SRshell_out[Int32(3), Int32(3), s, peak_id] = SRzz_tot
        end
        sync_threads()
        s += _I1
    end
    return
end

function Huntian.shell_gather_strain_gpu(delta_h::AbstractArray{<:Real,3},
                                           etax_h::AbstractArray{<:Real,3},
                                           etay_h::AbstractArray{<:Real,3},
                                           etaz_h::AbstractArray{<:Real,3},
                                           peaks_i::AbstractVector{<:Integer},
                                           peaks_j::AbstractVector{<:Integer},
                                           peaks_k::AbstractVector{<:Integer},
                                           stab::ShellTables;
                                           threads::Int=128)
    @assert size(delta_h) == size(etax_h) == size(etay_h) == size(etaz_h)
    @assert length(peaks_i) == length(peaks_j) == length(peaks_k)
    npeaks = length(peaks_i)
    nshells = stab.nshells

    delta_d = CuArray{Float32}(delta_h)
    etax_d  = CuArray{Float32}(etax_h)
    etay_d  = CuArray{Float32}(etay_h)
    etaz_d  = CuArray{Float32}(etaz_h)
    pi_d = CuArray{Int32}(peaks_i)
    pj_d = CuArray{Int32}(peaks_j)
    pk_d = CuArray{Int32}(peaks_k)
    stab_d = ShellTablesGPU(stab)

    Fshell_d  = CUDA.zeros(Float32, nshells, npeaks)
    nshell_d  = CUDA.zeros(Int32,   nshells, npeaks)
    Sshell_d  = CUDA.zeros(Float32, 3, nshells, npeaks)
    Gshell_d  = CUDA.zeros(Float32, 3, nshells, npeaks)
    SRshell_d = CUDA.zeros(Float32, 3, 3, nshells, npeaks)

    n1, n2, n3 = size(delta_d)
    shmem = 32 * sizeof(Float32) + 32 * sizeof(Int32)

    @cuda threads=threads blocks=npeaks shmem=shmem _shell_gather_strain_kernel!(
        Fshell_d, nshell_d, Sshell_d, Gshell_d, SRshell_d,
        delta_d, etax_d, etay_d, etaz_d,
        pi_d, pj_d, pk_d,
        stab_d.offsets_di, stab_d.offsets_dj, stab_d.offsets_dk,
        stab_d.shell_start, stab_d.shell_count,
        Int32(nshells), Int32(n1), Int32(n2), Int32(n3),
    )

    return Array(Fshell_d), Array(nshell_d), Array(Sshell_d),
           Array(Gshell_d), Array(SRshell_d)
end

# Full gather (port of the CPU gather in `analyse_peak_gpu`). Adds:
#   S2shell[L]   = Σ eta2_L[cell]              (2LPT; pass zeros to skip)
#   Gfshell[L]   = Σ delta[iv3, iv1, iv2] * d_L (transposed gradient)
#   lapdshell    = Σ lapd[cell]

function _shell_gather_full_kernel!(Fshell_out, nshell_out,
                                    Sshell_out, S2shell_out, Gshell_out,
                                    Gfshell_out, SRshell_out, lapdshell_out,
                                    delta, etax, etay, etaz,
                                    eta2x, eta2y, eta2z, lapd,
                                    peaks_i, peaks_j, peaks_k,
                                    off_di, off_dj, off_dk,
                                    shell_start, shell_count,
                                    nshells::Int32, n1::Int32, n2::Int32, n3::Int32,
                                    early::Int32, shell_r2, ir2min::Int32, fcrit_pp)
    peak_id = blockIdx().x
    tid = threadIdx().x
    bdim = blockDim().x

    sdata_f = CuDynamicSharedArray(Float32, 32)
    sdata_i = CuDynamicSharedArray(Int32, 32, 32 * sizeof(Float32))
    stop_flag = CuDynamicSharedArray(Int32, 1, 32 * sizeof(Float32) + 32 * sizeof(Int32))

    @inbounds ci = peaks_i[peak_id]
    @inbounds cj = peaks_j[peak_id]
    @inbounds ck = peaks_k[peak_id]

    # Early exit (early == 1): thread 1 replays _post_process_kernel!'s Fbar recurrence
    # shell by shell (same reduced sums, same Float32 expression) and m0 logic. Once
    # Fbar < fcrit·(1 − 1e-4) at some shell ≥ m0 (the first shell with r² > ir2min),
    # no quantity post-processing uses depends on shells beyond crossing radius + 2
    # (strain/gradient stencils are ±2 cells), so the gather stops at radius ≥ r_cross + 3.
    # The threshold margin makes the detected crossing never earlier than the true one;
    # unvisited shells keep their zero initialisation. Outputs are bit-identical.
    # Keep this recurrence in sync with _post_process_kernel!.
    if tid == _I1
        @inbounds stop_flag[1] = _I0
    end
    ee_fcrit = early == _I1 ? (@inbounds fcrit_pp[peak_id]) * (1.0f0 - 1.0f-4) : 0.0f0
    ee_Fbarp = 0.0f0; ee_dFbarp = 0.0f0; ee_radp = 0.0f0
    ee_past_m0 = false; ee_rstop = 3.4f38
    sync_threads()

    s = _I1
    while s <= nshells
        @inbounds s0 = shell_start[s]
        @inbounds nc = shell_count[s]

        local_F = 0.0f0
        local_Sx  = 0.0f0; local_Sy  = 0.0f0; local_Sz  = 0.0f0
        local_S2x = 0.0f0; local_S2y = 0.0f0; local_S2z = 0.0f0
        local_Gx  = 0.0f0; local_Gy  = 0.0f0; local_Gz  = 0.0f0
        local_Gfx = 0.0f0; local_Gfy = 0.0f0; local_Gfz = 0.0f0
        local_SRxx = 0.0f0; local_SRxy = 0.0f0; local_SRxz = 0.0f0
        local_SRyx = 0.0f0; local_SRyy = 0.0f0; local_SRyz = 0.0f0
        local_SRzx = 0.0f0; local_SRzy = 0.0f0; local_SRzz = 0.0f0
        local_lapd = 0.0f0
        local_n = _I0

        c = tid
        while c <= nc
            idx = s0 + c - _I1
            @inbounds di = off_di[idx]
            @inbounds dj = off_dj[idx]
            @inbounds dk = off_dk[idx]

            iv1 = ci + di
            iv2 = cj + dj
            iv3 = ck + dk

            if (_I1 <= iv1) & (iv1 <= n1) & (_I1 <= iv2) & (iv2 <= n2) & (_I1 <= iv3) & (iv3 <= n3)
                @inbounds d_val = delta[iv1, iv2, iv3]
                # Transposed read; in range only because the grid is cubic (n1=n2=n3).
                @inbounds df_val = delta[iv3, iv1, iv2]
                @inbounds ex = etax[iv1, iv2, iv3]
                @inbounds ey = etay[iv1, iv2, iv3]
                @inbounds ez = etaz[iv1, iv2, iv3]
                @inbounds e2x = eta2x[iv1, iv2, iv3]
                @inbounds e2y = eta2y[iv1, iv2, iv3]
                @inbounds e2z = eta2z[iv1, iv2, iv3]

                local_F += d_val
                local_Sx  += ex;  local_Sy  += ey;  local_Sz  += ez
                local_S2x += e2x; local_S2y += e2y; local_S2z += e2z

                fdi = Float32(di); fdj = Float32(dj); fdk = Float32(dk)
                local_Gx  += d_val  * fdi; local_Gy  += d_val  * fdj; local_Gz  += d_val  * fdk
                local_Gfx += df_val * fdi; local_Gfy += df_val * fdj; local_Gfz += df_val * fdk

                local_SRxx += ex * fdi; local_SRxy += ex * fdj; local_SRxz += ex * fdk
                local_SRyx += ey * fdi; local_SRyy += ey * fdj; local_SRyz += ey * fdk
                local_SRzx += ez * fdi; local_SRzy += ez * fdj; local_SRzz += ez * fdk

                @inbounds local_lapd += lapd[iv1, iv2, iv3]
                local_n += _I1
            end
            c += bdim
        end

        F_tot   = block_reduce_sum_f32(local_F,   sdata_f); sync_threads()
        Sx_tot  = block_reduce_sum_f32(local_Sx,  sdata_f); sync_threads()
        Sy_tot  = block_reduce_sum_f32(local_Sy,  sdata_f); sync_threads()
        Sz_tot  = block_reduce_sum_f32(local_Sz,  sdata_f); sync_threads()
        S2x_tot = block_reduce_sum_f32(local_S2x, sdata_f); sync_threads()
        S2y_tot = block_reduce_sum_f32(local_S2y, sdata_f); sync_threads()
        S2z_tot = block_reduce_sum_f32(local_S2z, sdata_f); sync_threads()
        Gx_tot  = block_reduce_sum_f32(local_Gx,  sdata_f); sync_threads()
        Gy_tot  = block_reduce_sum_f32(local_Gy,  sdata_f); sync_threads()
        Gz_tot  = block_reduce_sum_f32(local_Gz,  sdata_f); sync_threads()
        Gfx_tot = block_reduce_sum_f32(local_Gfx, sdata_f); sync_threads()
        Gfy_tot = block_reduce_sum_f32(local_Gfy, sdata_f); sync_threads()
        Gfz_tot = block_reduce_sum_f32(local_Gfz, sdata_f); sync_threads()
        SRxx_tot = block_reduce_sum_f32(local_SRxx, sdata_f); sync_threads()
        SRxy_tot = block_reduce_sum_f32(local_SRxy, sdata_f); sync_threads()
        SRxz_tot = block_reduce_sum_f32(local_SRxz, sdata_f); sync_threads()
        SRyx_tot = block_reduce_sum_f32(local_SRyx, sdata_f); sync_threads()
        SRyy_tot = block_reduce_sum_f32(local_SRyy, sdata_f); sync_threads()
        SRyz_tot = block_reduce_sum_f32(local_SRyz, sdata_f); sync_threads()
        SRzx_tot = block_reduce_sum_f32(local_SRzx, sdata_f); sync_threads()
        SRzy_tot = block_reduce_sum_f32(local_SRzy, sdata_f); sync_threads()
        SRzz_tot = block_reduce_sum_f32(local_SRzz, sdata_f); sync_threads()
        lapd_tot = block_reduce_sum_f32(local_lapd, sdata_f); sync_threads()
        n_tot    = block_reduce_sum_i32(local_n,   sdata_i)

        if tid == _I1
            @inbounds Fshell_out[s, peak_id] = F_tot
            @inbounds nshell_out[s, peak_id] = n_tot
            @inbounds Sshell_out[_I1, s, peak_id] = Sx_tot
            @inbounds Sshell_out[Int32(2), s, peak_id] = Sy_tot
            @inbounds Sshell_out[Int32(3), s, peak_id] = Sz_tot
            @inbounds S2shell_out[_I1, s, peak_id] = S2x_tot
            @inbounds S2shell_out[Int32(2), s, peak_id] = S2y_tot
            @inbounds S2shell_out[Int32(3), s, peak_id] = S2z_tot
            @inbounds Gshell_out[_I1, s, peak_id] = Gx_tot
            @inbounds Gshell_out[Int32(2), s, peak_id] = Gy_tot
            @inbounds Gshell_out[Int32(3), s, peak_id] = Gz_tot
            @inbounds Gfshell_out[_I1, s, peak_id] = Gfx_tot
            @inbounds Gfshell_out[Int32(2), s, peak_id] = Gfy_tot
            @inbounds Gfshell_out[Int32(3), s, peak_id] = Gfz_tot
            @inbounds SRshell_out[_I1, _I1, s, peak_id] = SRxx_tot
            @inbounds SRshell_out[_I1, Int32(2), s, peak_id] = SRxy_tot
            @inbounds SRshell_out[_I1, Int32(3), s, peak_id] = SRxz_tot
            @inbounds SRshell_out[Int32(2), _I1, s, peak_id] = SRyx_tot
            @inbounds SRshell_out[Int32(2), Int32(2), s, peak_id] = SRyy_tot
            @inbounds SRshell_out[Int32(2), Int32(3), s, peak_id] = SRyz_tot
            @inbounds SRshell_out[Int32(3), _I1, s, peak_id] = SRzx_tot
            @inbounds SRshell_out[Int32(3), Int32(2), s, peak_id] = SRzy_tot
            @inbounds SRshell_out[Int32(3), Int32(3), s, peak_id] = SRzz_tot
            @inbounds lapdshell_out[s, peak_id] = lapd_tot
            if early == _I1
                dF = n_tot > _I0 ? F_tot / Float32(n_tot) : 0.0f0
                if s == _I1
                    ee_Fbarp = dF; ee_dFbarp = dF; ee_radp = 0.0f0
                else
                    @inbounds r2s = shell_r2[s]
                    rad_s = sqrt(Float32(r2s))
                    rad3p = ee_radp * ee_radp * ee_radp
                    rad3 = rad_s * rad_s * rad_s
                    Fbar_s = (rad3p * ee_Fbarp + 0.5f0 * (ee_dFbarp + dF) * (rad3 - rad3p)) / rad3
                    ee_Fbarp = Fbar_s; ee_dFbarp = dF; ee_radp = rad_s
                    if !ee_past_m0 && Float32(r2s) > Float32(ir2min)
                        ee_past_m0 = true                       # this shell is m0
                    end
                    if ee_past_m0 && ee_rstop > 1.0f38 && Fbar_s < ee_fcrit
                        ee_rstop = rad_s + 3.0f0                # crossing found
                    end
                    if rad_s >= ee_rstop
                        @inbounds stop_flag[1] = _I1
                    end
                end
            end
        end
        sync_threads()
        @inbounds stop_flag[1] == _I1 && break
        s += _I1
    end
    return
end

# Shell early exit in the full gather (see _shell_gather_full_kernel!). Exact by
# construction; HUNTIAN_SHELL_EARLY_EXIT=0 disables it.
const _SHELL_EARLY_EXIT = Ref(get(ENV, "HUNTIAN_SHELL_EARLY_EXIT", "1") != "0")

# Launch the full gather on pre-allocated device arrays (no host round-trip).
# Early exit is enabled only when both shell_r2_d and fcrit_pp_d are given.
function _launch_shell_gather_full!(
        Fshell_d, nshell_d, Sshell_d, S2shell_d,
        Gshell_d, Gfshell_d, SRshell_d, lapdshell_d,
        delta_d, etax_d, etay_d, etaz_d,
        eta2x_d, eta2y_d, eta2z_d, lapd_d,
        pi_d, pj_d, pk_d,
        stab_d::ShellTablesGPU;
        threads::Int=128,
        shell_r2_d=nothing, ir2min::Integer=0, fcrit_pp_d=nothing)
    npeaks  = length(pi_d)
    nshells = stab_d.nshells
    n1, n2, n3 = size(delta_d)
    early = _SHELL_EARLY_EXIT[] && shell_r2_d !== nothing && fcrit_pp_d !== nothing
    r2_arg = early ? shell_r2_d : CUDA.zeros(Int32, 1)
    fc_arg = early ? fcrit_pp_d : CUDA.zeros(Float32, 1)
    shmem = 32 * sizeof(Float32) + 32 * sizeof(Int32) + sizeof(Int32)
    @cuda threads=threads blocks=npeaks shmem=shmem _shell_gather_full_kernel!(
        Fshell_d, nshell_d, Sshell_d, S2shell_d, Gshell_d, Gfshell_d, SRshell_d,
        lapdshell_d,
        delta_d, etax_d, etay_d, etaz_d, eta2x_d, eta2y_d, eta2z_d, lapd_d,
        pi_d, pj_d, pk_d,
        stab_d.offsets_di, stab_d.offsets_dj, stab_d.offsets_dk,
        stab_d.shell_start, stab_d.shell_count,
        Int32(nshells), Int32(n1), Int32(n2), Int32(n3),
        Int32(early), r2_arg, Int32(ir2min), fc_arg,
    )
    return nothing
end

function Huntian.shell_gather_full_gpu(delta_h::AbstractArray{<:Real,3},
                                         etax_h::AbstractArray{<:Real,3},
                                         etay_h::AbstractArray{<:Real,3},
                                         etaz_h::AbstractArray{<:Real,3},
                                         eta2x_h::AbstractArray{<:Real,3},
                                         eta2y_h::AbstractArray{<:Real,3},
                                         eta2z_h::AbstractArray{<:Real,3},
                                         peaks_i::AbstractVector{<:Integer},
                                         peaks_j::AbstractVector{<:Integer},
                                         peaks_k::AbstractVector{<:Integer},
                                         stab::ShellTables;
                                         threads::Int=128,
                                         lapd::Union{Nothing, AbstractArray{<:Real,3}}=nothing,
                                         return_lapd::Bool=false)
    @assert size(delta_h) == size(etax_h) == size(etay_h) == size(etaz_h) ==
            size(eta2x_h) == size(eta2y_h) == size(eta2z_h)
    @assert length(peaks_i) == length(peaks_j) == length(peaks_k)
    npeaks = length(peaks_i)
    nshells = stab.nshells

    delta_d = CuArray{Float32}(delta_h)
    etax_d  = CuArray{Float32}(etax_h)
    etay_d  = CuArray{Float32}(etay_h)
    etaz_d  = CuArray{Float32}(etaz_h)
    eta2x_d = CuArray{Float32}(eta2x_h)
    eta2y_d = CuArray{Float32}(eta2y_h)
    eta2z_d = CuArray{Float32}(eta2z_h)
    lapd_d = if lapd === nothing
        CUDA.zeros(Float32, size(delta_h)...)
    else
        @assert size(lapd) == size(delta_h)
        CuArray{Float32}(lapd)
    end
    pi_d = CuArray{Int32}(peaks_i)
    pj_d = CuArray{Int32}(peaks_j)
    pk_d = CuArray{Int32}(peaks_k)
    stab_d = ShellTablesGPU(stab)

    Fshell_d    = CUDA.zeros(Float32, nshells, npeaks)
    nshell_d    = CUDA.zeros(Int32,   nshells, npeaks)
    Sshell_d    = CUDA.zeros(Float32, 3, nshells, npeaks)
    S2shell_d   = CUDA.zeros(Float32, 3, nshells, npeaks)
    Gshell_d    = CUDA.zeros(Float32, 3, nshells, npeaks)
    Gfshell_d   = CUDA.zeros(Float32, 3, nshells, npeaks)
    SRshell_d   = CUDA.zeros(Float32, 3, 3, nshells, npeaks)
    lapdshell_d = CUDA.zeros(Float32, nshells, npeaks)

    _launch_shell_gather_full!(
        Fshell_d, nshell_d, Sshell_d, S2shell_d, Gshell_d, Gfshell_d, SRshell_d,
        lapdshell_d,
        delta_d, etax_d, etay_d, etaz_d, eta2x_d, eta2y_d, eta2z_d, lapd_d,
        pi_d, pj_d, pk_d, stab_d; threads=threads)

    # lapdshell is returned as an 8th value only with return_lapd=true.
    if return_lapd
        return (Array(Fshell_d), Array(nshell_d),
                Array(Sshell_d), Array(S2shell_d),
                Array(Gshell_d), Array(Gfshell_d),
                Array(SRshell_d), Array(lapdshell_d))
    else
        return (Array(Fshell_d), Array(nshell_d),
                Array(Sshell_d), Array(S2shell_d),
                Array(Gshell_d), Array(Gfshell_d),
                Array(SRshell_d))
    end
end

# Batch wrapper around `_eig3_symmetric_f32` (used for testing).

function _eig3_batch_kernel!(eigvals_out, mats, n::Int32)
    idx = (blockIdx().x - _I1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds a11 = mats[1, 1, idx]
        @inbounds a22 = mats[2, 2, idx]
        @inbounds a33 = mats[3, 3, idx]
        @inbounds a12 = 0.5f0 * (mats[1, 2, idx] + mats[2, 1, idx])
        @inbounds a13 = 0.5f0 * (mats[1, 3, idx] + mats[3, 1, idx])
        @inbounds a23 = 0.5f0 * (mats[2, 3, idx] + mats[3, 2, idx])
        e1, e2, e3 = _eig3_symmetric_f32(a11, a22, a33, a12, a13, a23)
        @inbounds eigvals_out[1, idx] = e1
        @inbounds eigvals_out[2, idx] = e2
        @inbounds eigvals_out[3, idx] = e3
    end
    return
end

function Huntian.eig3_symmetric_batch_gpu(mats_h::AbstractArray{<:Real,3};
                                            threads::Int=128)
    @assert size(mats_h, 1) == 3 && size(mats_h, 2) == 3
    n = size(mats_h, 3)
    mats_d = CuArray{Float32}(mats_h)
    out_d = CUDA.zeros(Float32, 3, n)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _eig3_batch_kernel!(out_d, mats_d, Int32(n))
    return Array(out_d)
end

# Batch wrapper around `_interp3_trilinear_f32` (used for testing).

function _interp3_batch_kernel!(out, table, xs, ys, zs, n::Int32,
                                 x1::Float32, y1::Float32, z1::Float32,
                                 dx_inv::Float32, dy_inv::Float32, dz_inv::Float32,
                                 nx::Int32, ny::Int32, nz::Int32,
                                 out_val::Float32)
    idx = (blockIdx().x - _I1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds x = xs[idx]
        @inbounds y = ys[idx]
        @inbounds z = zs[idx]
        v = _interp3_trilinear_f32(table, x, y, z, x1, y1, z1,
                                    dx_inv, dy_inv, dz_inv,
                                    nx, ny, nz, out_val)
        @inbounds out[idx] = v
    end
    return
end

function Huntian.interp3_trilinear_gpu(table_h::AbstractArray{<:Real,3},
                                         X1::Real, X2::Real,
                                         Y1::Real, Y2::Real,
                                         Z1::Real, Z2::Real,
                                         xs_h::AbstractVector{<:Real},
                                         ys_h::AbstractVector{<:Real},
                                         zs_h::AbstractVector{<:Real},
                                         out_val::Real=-1.0;
                                         threads::Int=128)
    @assert length(xs_h) == length(ys_h) == length(zs_h)
    n = length(xs_h)
    nx, ny, nz = size(table_h)

    table_d = CuArray{Float32}(table_h)
    xs_d = CuArray{Float32}(xs_h); ys_d = CuArray{Float32}(ys_h); zs_d = CuArray{Float32}(zs_h)
    out_d = CUDA.zeros(Float32, n)

    # axes are range(X1, X2, nx)
    dx_inv = Float32((nx - 1) / (X2 - X1))
    dy_inv = Float32((ny - 1) / (Y2 - Y1))
    dz_inv = Float32((nz - 1) / (Z2 - Z1))

    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _interp3_batch_kernel!(
        out_d, table_d, xs_d, ys_d, zs_d, Int32(n),
        Float32(X1), Float32(Y1), Float32(Z1),
        dx_inv, dy_inv, dz_inv,
        Int32(nx), Int32(ny), Int32(nz),
        Float32(out_val),
    )
    return Array(out_d)
end

# Single-thread wrapper around `_kernel_strain_f32` (used for testing).

function _kernel_strain_test_kernel!(out, rad, Gshell, Gshellf, SRshell, akk,
                                     mp::Int32, mlow::Int32, mupp::Int32,
                                     wRnor::Float32, aRnor::Float32,
                                     hlatt_1::Float32, hlatt_2::Float32)
    if threadIdx().x == _I1 && blockIdx().x == _I1
        (E11, E12, E13, E22, E23, E33, gx, gy, gz, gfx, gfy, gfz) =
            _kernel_strain_f32(rad, Gshell, Gshellf, SRshell, akk,
                                mp, mlow, mupp, wRnor, aRnor, hlatt_1, hlatt_2)
        @inbounds out[1]  = E11
        @inbounds out[2]  = E12
        @inbounds out[3]  = E13
        @inbounds out[4]  = E22
        @inbounds out[5]  = E23
        @inbounds out[6]  = E33
        @inbounds out[7]  = gx
        @inbounds out[8]  = gy
        @inbounds out[9]  = gz
        @inbounds out[10] = gfx
        @inbounds out[11] = gfy
        @inbounds out[12] = gfz
    end
    return
end

function Huntian.kernel_strain_gpu(rad_h::AbstractVector{<:Real},
                                     Gshell_h::AbstractMatrix{<:Real},
                                     Gshellf_h::AbstractMatrix{<:Real},
                                     SRshell_h::AbstractArray{<:Real,3},
                                     mp::Integer, mlow::Integer, mupp::Integer,
                                     wRnor::Real, aRnor::Real,
                                     hlatt_1::Real, hlatt_2::Real)
    @assert size(Gshell_h, 1) == 3 && size(Gshellf_h, 1) == 3
    @assert size(SRshell_h, 1) == 3 && size(SRshell_h, 2) == 3

    rad_d     = CuArray{Float32}(rad_h)
    Gshell_d  = CuArray{Float32}(Gshell_h)
    Gshellf_d = CuArray{Float32}(Gshellf_h)
    SRshell_d = CuArray{Float32}(SRshell_h)
    out_d     = CUDA.zeros(Float32, 12)

    @cuda threads=1 blocks=1 _kernel_strain_test_kernel!(
        out_d, rad_d, Gshell_d, Gshellf_d, SRshell_d, _AKK_GPU[],
        Int32(mp), Int32(mlow), Int32(mupp),
        Float32(wRnor), Float32(aRnor),
        Float32(hlatt_1), Float32(hlatt_2),
    )
    out = Array(out_d)
    Ebar = Float32[out[1] out[2] out[3];
                   out[2] out[4] out[5];
                   out[3] out[5] out[6]]
    grad  = Float32[out[7],  out[8],  out[9]]
    gradf = Float32[out[10], out[11], out[12]]
    return (Ebar, grad, gradf)
end

# Post-process kernel, second stage after `_shell_gather_full_kernel!`.
# One block per peak; only thread 1 works, sequentially, on the peak's
# normalised profiles staged in shared memory: fcrit crossing, inward
# strain/zvir walk, RTHL interpolation, Sbar/Srb/d2F, zvir_half, mask update.
# RTHL = -1 signals no_collapse (as on CPU).

# Upper bound on distinct r² shells per peak (shared-memory sizing).
# nhunt=23 → 443 shells, nhunt=25 → 523.
const _MAX_SHELLS_GPU = Int32(512)

# Exceeding _MAX_SHELLS_GPU overflows the shared-memory profiles silently
# (most peaks then fail to collapse), so reject it up front.
_check_shell_capacity(nshells::Integer) = nshells <= _MAX_SHELLS_GPU || error(
    "GPU shell analysis supports at most $(_MAX_SHELLS_GPU) radial shells but the hunt radius gives " *
    "$(nshells): use nbuff ≤ 25 (nhunt = min(nbuff−1, 1.75·Rf_max/a) ≤ 24) or the CPU path")

# Mirrors `RadialShell.no_collapse()`: all zeros except RTHL = zvir_half = -1.
@inline function _write_no_collapse_outputs!(
        RTHL_out, Fbarx_out, e_v_out, p_v_out,
        strain_final_out, eigs_out,
        Srb_out, Sbar_out, Sbar2_out,
        gradpk_out, gradpkf_out, gradpkrf_out, d2F_out,
        zvir_half_out,
        peak_id)
    @inbounds RTHL_out[peak_id]  = -1.0f0
    @inbounds Fbarx_out[peak_id] = 0.0f0
    @inbounds e_v_out[peak_id]   = 0.0f0
    @inbounds p_v_out[peak_id]   = 0.0f0
    @inbounds Srb_out[peak_id]   = 0.0f0
    @inbounds d2F_out[peak_id]   = 0.0f0
    @inbounds zvir_half_out[peak_id] = -1.0f0
    for L in _I1:Int32(3)
        @inbounds Sbar_out[L, peak_id]     = 0.0f0
        @inbounds Sbar2_out[L, peak_id]    = 0.0f0
        @inbounds gradpk_out[L, peak_id]   = 0.0f0
        @inbounds gradpkf_out[L, peak_id]  = 0.0f0
        @inbounds gradpkrf_out[L, peak_id] = 0.0f0
        @inbounds eigs_out[L, peak_id]     = 0.0f0
        for K in _I1:Int32(3)
            @inbounds strain_final_out[L, K, peak_id] = 0.0f0
        end
    end
    return
end

function _post_process_kernel!(
    # Outputs (indexed by peak)
    RTHL_out,           # Vector{Float32} (npeaks)
    Fbarx_out,          # Vector{Float32}
    e_v_out,            # Vector{Float32}
    p_v_out,            # Vector{Float32}
    strain_final_out,   # Array{Float32,3} (3, 3, npeaks)
    eigs_out,           # Array{Float32,2} (3, npeaks)
    Srb_out,            # Vector{Float32}
    Sbar_out,           # Array{Float32,2} (3, npeaks)
    Sbar2_out,          # Array{Float32,2} (3, npeaks)
    gradpk_out,         # Array{Float32,2} (3, npeaks)
    gradpkf_out,        # Array{Float32,2} (3, npeaks)
    gradpkrf_out,       # Array{Float32,2} (3, npeaks)
    d2F_out,            # Vector{Float32} — Laplacian average within RTHL
    zvir_half_out,      # Vector{Float32} — formation redshift (collapse z at RTHL/2)
    mask_out,           # Array{Int8,3} (n1, n2, n3) — side-effect, cells within RTHL → 1
    # Inputs from gather (un-normalised)
    Fshell, nshell_in, Sshell_in, S2shell_in, Gshell_raw, Gfshell_raw, SRshell_raw,
    lapdshell_in,
    # Peak centers for mask update
    peaks_i, peaks_j, peaks_k,
    # Shell offset tables (also for mask update)
    off_di, off_dj, off_dk, shell_start, shell_count,
    # Shell geometry
    shell_r2,
    # Collapse table
    ct_table, ct_X1::Float32, ct_Y1::Float32, ct_Z1::Float32,
    ct_dxi::Float32, ct_dyi::Float32, ct_dzi::Float32,
    ct_nx::Int32, ct_ny::Int32, ct_nz::Int32, ct_out_val::Float32,
    # Kernel tables & scaling
    akk_tab,
    # Per-peak collapse state: ZZon_pp[i] = 1+z_collapse, fcrit_pp[i] the
    # corresponding Fbar threshold. Broadcast scalars for ievol==0; per-peak
    # lightcone redshift for ievol==1.
    ZZon_pp, fcrit_pp, Rfclvi_r2::Float32,
    wRnor::Float32, aRnor::Float32, hlatt_1::Float32, hlatt_2::Float32,
    ir2min::Int32,
    nshells::Int32, nshells_max::Int32,
    n1::Int32, n2::Int32, n3::Int32, nbuff::Int32, update_mask::Bool)

    peak_id = blockIdx().x
    @inbounds ZZon  = ZZon_pp[peak_id]
    @inbounds fcrit = fcrit_pp[peak_id]

    # Sized for _MAX_SHELLS_GPU; only the first nshells_max slots are used.
    shmem_rad   = CuDynamicSharedArray(Float32, _MAX_SHELLS_GPU, 0)
    shmem_Fbar  = CuDynamicSharedArray(Float32, _MAX_SHELLS_GPU,
                                        Int(_MAX_SHELLS_GPU) * sizeof(Float32))
    shmem_Gn    = CuDynamicSharedArray(Float32, (Int32(3), _MAX_SHELLS_GPU),
                                        Int(_MAX_SHELLS_GPU) * sizeof(Float32) * 2)
    shmem_Gfn   = CuDynamicSharedArray(Float32, (Int32(3), _MAX_SHELLS_GPU),
                                        Int(_MAX_SHELLS_GPU) * sizeof(Float32) * (2 + 3))
    shmem_SRn   = CuDynamicSharedArray(Float32, (Int32(3), Int32(3), _MAX_SHELLS_GPU),
                                        Int(_MAX_SHELLS_GPU) * sizeof(Float32) * (2 + 3 + 3))

    tid = threadIdx().x
    if tid != _I1
        return
    end

    # Normalised per-shell profile. Shell 1 is the centre (r²=0): rad=0, zero G/Gf/SR.
    @inbounds n1_cnt = nshell_in[1, peak_id]
    @inbounds F1_sum = Fshell[1, peak_id]
    dFbar_1 = n1_cnt > _I0 ? F1_sum / Float32(n1_cnt) : 0.0f0
    @inbounds shmem_rad[1]  = 0.0f0
    @inbounds shmem_Fbar[1] = dFbar_1
    @inbounds shmem_Gn[_I1, 1]  = 0.0f0
    @inbounds shmem_Gn[Int32(2), 1] = 0.0f0
    @inbounds shmem_Gn[Int32(3), 1] = 0.0f0
    @inbounds shmem_Gfn[_I1, 1] = 0.0f0
    @inbounds shmem_Gfn[Int32(2), 1] = 0.0f0
    @inbounds shmem_Gfn[Int32(3), 1] = 0.0f0
    @inbounds shmem_SRn[_I1, _I1, 1] = 0.0f0
    @inbounds shmem_SRn[_I1, Int32(2), 1] = 0.0f0
    @inbounds shmem_SRn[_I1, Int32(3), 1] = 0.0f0
    @inbounds shmem_SRn[Int32(2), _I1, 1] = 0.0f0
    @inbounds shmem_SRn[Int32(2), Int32(2), 1] = 0.0f0
    @inbounds shmem_SRn[Int32(2), Int32(3), 1] = 0.0f0
    @inbounds shmem_SRn[Int32(3), _I1, 1] = 0.0f0
    @inbounds shmem_SRn[Int32(3), Int32(2), 1] = 0.0f0
    @inbounds shmem_SRn[Int32(3), Int32(3), 1] = 0.0f0

    dFbarp = dFbar_1
    s = Int32(2)
    while s <= nshells_max
        @inbounds r2 = shell_r2[s]
        rad_s = sqrt(Float32(r2))
        @inbounds shmem_rad[s] = rad_s
        rad_s_inv = 1.0f0 / rad_s
        @inbounds n_s = nshell_in[s, peak_id]
        @inbounds F_s = Fshell[s, peak_id]
        dFbar_s = n_s > _I0 ? F_s / Float32(n_s) : 0.0f0
        @inbounds rad_sp = shmem_rad[s - _I1]
        rad3p = rad_sp * rad_sp * rad_sp
        rad3  = rad_s * rad_s * rad_s
        @inbounds Fbarp = shmem_Fbar[s - _I1]
        Fbar_s = (rad3p * Fbarp + 0.5f0 * (dFbarp + dFbar_s) * (rad3 - rad3p)) / rad3
        @inbounds shmem_Fbar[s] = Fbar_s
        dFbarp = dFbar_s
        # Normalise G, Gf, SR by rad_s (CPU: Gshell[L][m] = local_G[L] / rad[m])
        for L in _I1:Int32(3)
            @inbounds shmem_Gn[L, s]  = Gshell_raw[L, s, peak_id]  * rad_s_inv
            @inbounds shmem_Gfn[L, s] = Gfshell_raw[L, s, peak_id] * rad_s_inv
            for K in _I1:Int32(3)
                @inbounds shmem_SRn[L, K, s] = SRshell_raw[L, K, s, peak_id] * rad_s_inv
            end
        end
        s += _I1
    end
    m = nshells_max

    # Gradient at filter scale (gradpkrf). mrf = shell closest to Rfclvi/alatt
    # in r². Replicates the CPU analyse_peak_gpu mrfi counter, which lags the
    # shell index by one (mrfi == s-1, since shells have unique r²).
    mrf = _I1
    rmrf = 10.0f0
    mrfi = _I1
    s_mrf = Int32(2)
    while s_mrf <= nshells_max
        @inbounds r2_s = shell_r2[s_mrf]
        dist_mrf = abs(Float32(r2_s) - Rfclvi_r2)
        if dist_mrf < rmrf
            mrf = mrfi
            rmrf = dist_mrf
        end
        if s_mrf < nshells_max
            mrfi += _I1
        end
        s_mrf += _I1
    end
    @inbounds rlow_rf = max(shmem_rad[mrf] - 2.0f0, 0.0f0)
    mlow_rf = mrf
    if mrf > _I1
        m1_rf = mrf - _I1
        while m1_rf >= _I1
            @inbounds if shmem_rad[m1_rf] > rlow_rf
                mlow_rf = m1_rf
            end
            m1_rf -= _I1
        end
    end
    @inbounds rupp_rf = shmem_rad[mrf] + 2.0f0
    mupp_rf = mrf
    m1_rf = mrf + _I1
    while m1_rf <= m
        @inbounds if shmem_rad[m1_rf] < rupp_rf
            mupp_rf = m1_rf
        end
        m1_rf += _I1
    end
    # gradpkrf is the grad (not gradf) output of kernel_strain at mrf.
    (_, _, _, _, _, _,
     gradpkrf_x, gradpkrf_y, gradpkrf_z, _, _, _) = _kernel_strain_f32(
        shmem_rad, shmem_Gn, shmem_Gfn, shmem_SRn, akk_tab,
        mrf, mlow_rf, mupp_rf, wRnor, aRnor, hlatt_1, hlatt_2)

    # Initial m0/mupp, as tracked in the CPU gather loop.
    m0 = _I1
    ifcrit = _I1
    rupp_init = sqrt(Float32(ir2min)) + 2.0f0
    mupp = _I1
    s = Int32(2)
    while s <= nshells_max
        @inbounds r2 = shell_r2[s]
        if ifcrit == _I1
            m0 = s
            if Float32(r2) > Float32(ir2min)
                @inbounds rupp_init = shmem_rad[m0] + 2.0f0
                ifcrit = _I0
                mupp = s
            end
        else
            @inbounds if shmem_rad[s] < rupp_init
                mupp = s
            end
        end
        s += _I1
    end

    # Locate the fcrit crossing near m0.
    @inbounds Fbar_m0 = shmem_Fbar[m0]
    found_cross = false
    if Fbar_m0 >= fcrit
        mm = m0
        while mm <= m
            @inbounds if shmem_Fbar[mm] < fcrit
                m0 = mm
                found_cross = true
                break
            end
            mm += _I1
        end
        if !found_cross
            mupp = m
        end
        if found_cross
            @inbounds rupp_m0 = shmem_rad[m0] + 2.0f0
            mupp = m0
            mm = m0 + _I1
            while mm <= m
                @inbounds if shmem_rad[mm] < rupp_m0
                    mupp = mm
                end
                mm += _I1
            end
        end
    else
        mstart = m0 > _I1 ? (m0 - _I1) : _I1
        mp = mstart
        while mp >= _I1
            @inbounds if shmem_Fbar[mp] >= fcrit
                found_cross = true
                break
            end
            m0 = mp
            mp -= _I1
        end
        if !found_cross
            _write_no_collapse_outputs!(
                RTHL_out, Fbarx_out, e_v_out, p_v_out,
                strain_final_out, eigs_out,
                Srb_out, Sbar_out, Sbar2_out,
                gradpk_out, gradpkf_out, gradpkrf_out, d2F_out,
                zvir_half_out, peak_id)
            return
        end
        @inbounds rupp_m0 = shmem_rad[m0] + 2.0f0
        mupp_new = m0
        mm = m0 + _I1
        while mm <= m
            @inbounds if shmem_rad[mm] < rupp_m0
                mupp_new = mm
            end
            mm += _I1
        end
        mupp = mupp_new
    end

    # Strain at m0, then walk inward until zvir >= ZZon.
    mlow = m0
    @inbounds rlow = max(shmem_rad[m0] - 2.0f0, 0.0f0)
    if m0 > _I1
        m1 = m0 - _I1
        while m1 >= _I1
            @inbounds if shmem_rad[m1] > rlow
                mlow = m1
            end
            m1 -= _I1
        end
    end

    (E11_m0, E12_m0, E13_m0, E22_m0, E23_m0, E33_m0,
     gx_m0, gy_m0, gz_m0, gfx_m0, gfy_m0, gfz_m0) = _kernel_strain_f32(
        shmem_rad, shmem_Gn, shmem_Gfn, shmem_SRn, akk_tab,
        m0, mlow, mupp, wRnor, aRnor, hlatt_1, hlatt_2)
    (lam1_m0, lam2_m0, lam3_m0) = _eig3_symmetric_f32(
        E11_m0, E22_m0, E33_m0, E12_m0, E13_m0, E23_m0)
    Frho_m0_raw = lam1_m0 + lam2_m0 + lam3_m0
    e_v_m0 = Frho_m0_raw > 0.0f0 ? 0.5f0 * (lam3_m0 - lam1_m0) / Frho_m0_raw : 0.0f0
    p_v_m0 = Frho_m0_raw > 0.0f0 ? 0.5f0 * (lam3_m0 + lam1_m0 - 2.0f0 * lam2_m0) / Frho_m0_raw : 0.0f0
    Frhoh_m0 = Frho_m0_raw
    @inbounds Frho_m0_fbar = shmem_Fbar[m0]
    # Initial zvir1p = -1 as on CPU (Fortran leaves it uninitialized).
    zvir1p_m0 = -1.0f0

    Frhoh = Frhoh_m0
    Frho_val = Frho_m0_fbar
    e_v_curr = e_v_m0
    p_v_curr = p_v_m0
    E11_last = E11_m0; E12_last = E12_m0; E13_last = E13_m0
    E22_last = E22_m0; E23_last = E23_m0; E33_last = E33_m0
    # *_prev: values at the last non-collapsing shell
    E11_prev = E11_m0; E12_prev = E12_m0; E13_prev = E13_m0
    E22_prev = E22_m0; E23_prev = E23_m0; E33_prev = E33_m0
    gx_last = gx_m0;  gy_last = gy_m0;  gz_last = gz_m0
    gfx_last = gfx_m0; gfy_last = gfy_m0; gfz_last = gfz_m0
    gx_prev = gx_m0;  gy_prev = gy_m0;  gz_prev = gz_m0
    gfx_prev = gfx_m0; gfy_prev = gfy_m0; gfz_prev = gfz_m0
    Frhpk = Frhoh_m0; Fnupk = Frho_m0_fbar; Fevpk = e_v_m0; Fpvpk = p_v_m0

    collapsed = false
    zvir1 = -1.0f0
    zvir1p = zvir1p_m0

    if zvir1p_m0 >= ZZon
        collapsed = true
    else
        zvir1 = zvir1p_m0
        if m0 == _I1
            _write_no_collapse_outputs!(
                RTHL_out, Fbarx_out, e_v_out, p_v_out,
                strain_final_out, eigs_out,
                Srb_out, Sbar_out, Sbar2_out,
                gradpk_out, gradpkf_out, gradpkrf_out, d2F_out,
                zvir_half_out, peak_id)
            return
        end

        mp = m0 - _I1
        while mp >= _I1
            @inbounds rupp_mp = shmem_rad[mp] + 2.0f0
            @inbounds rlow_mp = max(shmem_rad[mp] - 2.0f0, 0.0f0)

            @inbounds if shmem_rad[mupp] > rupp_mp
                mupp_new = mp
                m1 = mp + _I1
                while m1 <= mupp
                    @inbounds if shmem_rad[m1] < rupp_mp
                        mupp_new = m1
                    end
                    m1 += _I1
                end
                mupp = mupp_new
            end
            @inbounds if mlow > _I1 && shmem_rad[mlow - _I1] > rlow_mp
                mlownew = mlow
                m1 = mlow - _I1
                while m1 >= _I1
                    @inbounds if shmem_rad[m1] > rlow_mp
                        mlownew = m1
                    end
                    m1 -= _I1
                end
                mlow = mlownew
            end

            (E11_mp, E12_mp, E13_mp, E22_mp, E23_mp, E33_mp,
             gx_mp, gy_mp, gz_mp, gfx_mp, gfy_mp, gfz_mp) = _kernel_strain_f32(
                shmem_rad, shmem_Gn, shmem_Gfn, shmem_SRn, akk_tab,
                mp, mlow, mupp, wRnor, aRnor, hlatt_1, hlatt_2)
            (lam1_mp, lam2_mp, lam3_mp) = _eig3_symmetric_f32(
                E11_mp, E22_mp, E33_mp, E12_mp, E13_mp, E23_mp)
            Frho_mp_raw = lam1_mp + lam2_mp + lam3_mp
            e_v_mp = Frho_mp_raw > 0.0f0 ? 0.5f0 * (lam3_mp - lam1_mp) / Frho_mp_raw : 0.0f0
            p_v_mp = Frho_mp_raw > 0.0f0 ? 0.5f0 * (lam3_mp + lam1_mp - 2.0f0 * lam2_mp) / Frho_mp_raw : 0.0f0

            Frhoh_mp = Frho_mp_raw
            @inbounds Frho_mp_fbar = shmem_Fbar[mp]

            zvir1p_mp = -1.0f0
            if Frho_mp_fbar > 0.0f0
                poe_mp = e_v_mp < 1.0f-5 ? 0.0f0 : p_v_mp / e_v_mp
                zvir1p_mp = _interp3_trilinear_f32(
                    ct_table, log10(Frho_mp_fbar), e_v_mp, poe_mp,
                    ct_X1, ct_Y1, ct_Z1, ct_dxi, ct_dyi, ct_dzi,
                    ct_nx, ct_ny, ct_nz, ct_out_val)
            end

            if zvir1p_mp >= ZZon
                m0 = mp + _I1
                zvir1p = zvir1p_mp
                E11_last = E11_mp; E12_last = E12_mp; E13_last = E13_mp
                E22_last = E22_mp; E23_last = E23_mp; E33_last = E33_mp
                gx_last = gx_mp;   gy_last = gy_mp;   gz_last = gz_mp
                gfx_last = gfx_mp; gfy_last = gfy_mp; gfz_last = gfz_mp
                Frhoh = Frhoh_mp
                Frho_val = Frho_mp_fbar
                e_v_curr = e_v_mp
                p_v_curr = p_v_mp
                collapsed = true
                break
            end

            zvir1 = zvir1p_mp
            Frhpk = Frhoh_mp; Fnupk = Frho_mp_fbar; Fevpk = e_v_mp; Fpvpk = p_v_mp
            E11_prev = E11_mp; E12_prev = E12_mp; E13_prev = E13_mp
            E22_prev = E22_mp; E23_prev = E23_mp; E33_prev = E33_mp
            gx_prev = gx_mp;   gy_prev = gy_mp;   gz_prev = gz_mp
            gfx_prev = gfx_mp; gfy_prev = gfy_mp; gfz_prev = gfz_mp
            Frhoh = Frhoh_mp
            Frho_val = Frho_mp_fbar
            e_v_curr = e_v_mp
            p_v_curr = p_v_mp
            E11_last = E11_mp; E12_last = E12_mp; E13_last = E13_mp
            E22_last = E22_mp; E23_last = E23_mp; E33_last = E33_mp
            gx_last = gx_mp;   gy_last = gy_mp;   gz_last = gz_mp
            gfx_last = gfx_mp; gfy_last = gfy_mp; gfz_last = gfz_mp
            mp -= _I1
        end
    end

    if !collapsed
        _write_no_collapse_outputs!(
            RTHL_out, Fbarx_out, e_v_out, p_v_out,
            strain_final_out, eigs_out,
            Srb_out, Sbar_out, Sbar2_out,
            gradpk_out, gradpkf_out, gradpkrf_out, d2F_out,
            zvir_half_out, peak_id)
        return
    end

    # RTHL: interpolate in r^3 to where zvir crosses ZZon.
    dZvir = zvir1p - zvir1
    RTHL = 0.0f0
    Fbarx = 0.0f0
    frac = 0.0f0
    if zvir1 > 0.0f0 && dZvir != 0.0f0
        @inbounds radmp = shmem_rad[m0 - _I1]
        @inbounds radm  = shmem_rad[m0]
        RTHL3 = radmp * radmp * radmp +
                (radm * radm * radm - radmp * radmp * radmp) * (zvir1p - ZZon) / dZvir
        RTHL = cbrt(RTHL3)
    else
        @inbounds RTHL = shmem_rad[m0 - _I1]
    end

    if RTHL <= 0.0f0
        _write_no_collapse_outputs!(
            RTHL_out, Fbarx_out, e_v_out, p_v_out,
            strain_final_out, eigs_out,
            Srb_out, Sbar_out, Sbar2_out,
            gradpk_out, gradpkf_out, gradpkrf_out, d2F_out,
            zvir_half_out, peak_id)
        return
    end

    @inbounds rad3p = shmem_rad[m0 - _I1] * shmem_rad[m0 - _I1] * shmem_rad[m0 - _I1]
    @inbounds rad3  = shmem_rad[m0] * shmem_rad[m0] * shmem_rad[m0]
    drad3 = rad3 - rad3p
    RTHL3 = RTHL * RTHL * RTHL
    @inbounds dFbar = shmem_Fbar[m0] - shmem_Fbar[m0 - _I1]

    if zvir1 > 0.0f0 && drad3 != 0.0f0
        frac = (RTHL3 - rad3p) / drad3
        @inbounds Fbarx = shmem_Fbar[m0 - _I1] + frac * dFbar
        Frhpk = Frhoh + frac * (Frhpk - Frhoh)
        Fnupk = Frho_val + frac * (Fnupk - Frho_val)
        Fevpk = e_v_curr + frac * (Fevpk - e_v_curr)
        Fpvpk = p_v_curr + frac * (Fpvpk - p_v_curr)
        E11_mat = E11_last + frac * (E11_prev - E11_last)
        E12_mat = E12_last + frac * (E12_prev - E12_last)
        E13_mat = E13_last + frac * (E13_prev - E13_last)
        E22_mat = E22_last + frac * (E22_prev - E22_last)
        E23_mat = E23_last + frac * (E23_prev - E23_last)
        E33_mat = E33_last + frac * (E33_prev - E33_last)
        gx_mat  = gx_last  + frac * (gx_prev  - gx_last)
        gy_mat  = gy_last  + frac * (gy_prev  - gy_last)
        gz_mat  = gz_last  + frac * (gz_prev  - gz_last)
        gfx_mat = gfx_last + frac * (gfx_prev - gfx_last)
        gfy_mat = gfy_last + frac * (gfy_prev - gfy_last)
        gfz_mat = gfz_last + frac * (gfz_prev - gfz_last)
    else
        @inbounds Fbarx = shmem_Fbar[m0 - _I1]
        Fnupk = Frho_val; Fevpk = e_v_curr; Fpvpk = p_v_curr
        E11_mat = E11_last; E12_mat = E12_last; E13_mat = E13_last
        E22_mat = E22_last; E23_mat = E23_last; E33_mat = E33_last
        gx_mat  = gx_prev;  gy_mat  = gy_prev;  gz_mat  = gz_prev
        gfx_mat = gfx_prev; gfy_mat = gfy_prev; gfz_mat = gfz_prev
    end

    # Rescale interpolated strain so its trace equals Fbarx.
    tr_E = E11_mat + E22_mat + E33_mat
    if tr_E != 0.0f0
        sc = Fbarx / tr_E
        E11n = E11_mat * sc; E12n = E12_mat * sc; E13n = E13_mat * sc
        E22n = E22_mat * sc; E23n = E23_mat * sc; E33n = E33_mat * sc
    else
        E11n = E11_mat; E12n = E12_mat; E13n = E13_mat
        E22n = E22_mat; E23n = E23_mat; E33n = E33_mat
    end
    (lamf1, lamf2, lamf3) = _eig3_symmetric_f32(E11n, E22n, E33n, E12n, E13n, E23n)
    Frhoc = lamf1 + lamf2 + lamf3
    e_v_f = Frhoc > 0.0f0 ? 0.5f0 * (lamf3 - lamf1) / Frhoc : 0.0f0
    p_v_f = Frhoc > 0.0f0 ? 0.5f0 * (lamf3 + lamf1 - 2.0f0 * lamf2) / Frhoc : 0.0f0

    # Final strain = Ebar_last * (Fbarx / Frhoh), as on CPU (uses E_last, not the interpolated E).
    if Frhoh != 0.0f0
        sc2 = Fbarx / Frhoh
        E11f = E11_last * sc2; E12f = E12_last * sc2; E13f = E13_last * sc2
        E22f = E22_last * sc2; E23f = E23_last * sc2; E33f = E33_last * sc2
    else
        E11f = E11_last; E12f = E12_last; E13f = E13_last
        E22f = E22_last; E23f = E23_last; E33f = E33_last
    end

    # Sbar, Sbar2: mean displacement over in-bounds cells of shells 1..m0-1
    # (raw sums read from global memory; not staged in shared memory).
    Sbar_x  = 0.0f0; Sbar_y  = 0.0f0; Sbar_z  = 0.0f0
    Sbar2_x = 0.0f0; Sbar2_y = 0.0f0; Sbar2_z = 0.0f0
    nSbar = _I0
    m1 = _I1
    while m1 <= (m0 - _I1)
        @inbounds n_m1 = nshell_in[m1, peak_id]
        if n_m1 > _I0
            @inbounds Sbar_x += Sshell_in[_I1, m1, peak_id]
            @inbounds Sbar_y += Sshell_in[Int32(2), m1, peak_id]
            @inbounds Sbar_z += Sshell_in[Int32(3), m1, peak_id]
            @inbounds Sbar2_x += S2shell_in[_I1, m1, peak_id]
            @inbounds Sbar2_y += S2shell_in[Int32(2), m1, peak_id]
            @inbounds Sbar2_z += S2shell_in[Int32(3), m1, peak_id]
            nSbar += n_m1
        end
        m1 += _I1
    end
    if nSbar > _I0
        invN = 1.0f0 / Float32(nSbar)
        Sbar_x *= invN; Sbar_y *= invN; Sbar_z *= invN
        Sbar2_x *= invN; Sbar2_y *= invN; Sbar2_z *= invN
    end

    # Srb: ∫ Fbar(r) r^4 dr over 0..RTHL, normalised by Fbarx * RTHL^5
    # (trapezoidal in r^5).
    Srb = 0.0f0
    rad5p = 0.0f0
    if m0 > Int32(2)
        mp = Int32(2)
        while mp <= (m0 - _I1)
            @inbounds rad_mp = shmem_rad[mp]
            rad5 = rad_mp * rad_mp * rad_mp * rad_mp * rad_mp
            @inbounds Srb += 0.5f0 * (shmem_Fbar[mp - _I1] + shmem_Fbar[mp]) * (rad5 - rad5p)
            rad5p = rad5
            mp += _I1
        end
    end
    RTHL5 = RTHL3 * RTHL * RTHL
    if zvir1 > 0.0f0 && dZvir != 0.0f0
        @inbounds Srb += 0.5f0 * (shmem_Fbar[m0 - _I1] + Fbarx) * (RTHL5 - rad5p)
    end
    denom_Srb = Fbarx * RTHL5
    if denom_Srb > 0.0f0
        Srb /= denom_Srb
    end

    # d2F: mean lapd over in-bounds cells of shells with r ≤ RTHL.
    d2F = 0.0f0
    nd2 = _I0
    s_lap = _I1
    while s_lap <= nshells
        @inbounds r2_s = shell_r2[s_lap]
        sqrt_r2 = sqrt(Float32(r2_s))
        if sqrt_r2 > RTHL
            break
        end
        @inbounds d2F += lapdshell_in[s_lap, peak_id]
        @inbounds nd2 += nshell_in[s_lap, peak_id]
        s_lap += _I1
    end
    if nd2 > _I0
        d2F /= Float32(nd2)
    end

    # zvir_half (formation redshift, as in RadialShell.analyse_peak): max over
    # 10 radii rcur = RTHL * cbrt(0.5..1) of the collapse-table z at rcur,
    # using kernel_strain with hlatt = 1. Only for RTHL ≥ 3.
    zvir_half = -1.0f0
    if RTHL >= 3.0f0
        jj_int = Int32(1)
        while jj_int <= Int32(10)
            tfrac = (Float32(jj_int - _I1)) * (1.0f0 / 9.0f0) * 0.5f0 + 0.5f0
            rcur = RTHL * cbrt(tfrac)

            # mupp_rc: first m1 in 2..nshells_max-1 with rad[m1] > rcur+2
            rupp_rc = rcur + 2.0f0
            mupp_rc = _I1
            m1_u = Int32(2)
            while m1_u <= nshells_max - _I1
                @inbounds r_u = shmem_rad[m1_u]
                if r_u > rupp_rc
                    mupp_rc = m1_u
                    break
                elseif r_u == 0.0f0
                    mupp_rc = m1_u - _I1
                    break
                end
                mupp_rc = m1_u
                m1_u += _I1
            end

            # mlow_rc: first m1 in 1..nshells_max-1 with rad[m1] > rcur-2
            rlow_rc = rcur - 2.0f0
            if rlow_rc < 0.0f0
                rlow_rc = 0.0f0
            end
            mlow_rc = _I1
            m1_l = _I1
            while m1_l <= nshells_max - _I1
                @inbounds if shmem_rad[m1_l] > rlow_rc
                    mlow_rc = m1_l
                    break
                end
                m1_l += _I1
            end

            # m0_rc: first m1 with rad[m1] > rcur, else nshells_max-1 (as on CPU)
            m0_rc = _I1
            m1_0 = _I1
            while m1_0 <= nshells_max - _I1
                @inbounds if shmem_rad[m1_0] > rcur
                    m0_rc = m1_0
                    break
                end
                m0_rc = m1_0
                m1_0 += _I1
            end

            (E11_rc, E12_rc, E13_rc, E22_rc, E23_rc, E33_rc,
             _, _, _, _, _, _) = _kernel_strain_f32(
                shmem_rad, shmem_Gn, shmem_Gfn, shmem_SRn, akk_tab,
                m0_rc, mlow_rc, mupp_rc, wRnor, aRnor, 1.0f0, 1.0f0)

            (lam1_rc, lam2_rc, lam3_rc) = _eig3_symmetric_f32(
                E11_rc, E22_rc, E33_rc, E12_rc, E13_rc, E23_rc)
            Frho_rc = lam1_rc + lam2_rc + lam3_rc
            e_v_rc = 0.0f0; p_v_rc = 0.0f0
            if Frho_rc > 0.0f0
                e_v_rc = 0.5f0 * (lam3_rc - lam1_rc) / Frho_rc
                p_v_rc = 0.5f0 * (lam3_rc + lam1_rc - 2.0f0 * lam2_rc) / Frho_rc
            end

            @inbounds Frhoc_rc = shmem_Fbar[m0_rc]
            poe_rc = e_v_rc < 1.0f-5 ? 0.0f0 : p_v_rc / e_v_rc
            zvir_rc = -1.0f0
            # No eigensolver-failure flag: CPU iflag_rc is always 0 with LAPACK.
            if Frhoc_rc > 0.0f0
                zvir_rc = _interp3_trilinear_f32(
                    ct_table, log10(Frhoc_rc), e_v_rc, poe_rc,
                    ct_X1, ct_Y1, ct_Z1, ct_dxi, ct_dyi, ct_dzi,
                    ct_nx, ct_ny, ct_nz, ct_out_val)
            end
            zrc_shifted = zvir_rc - 1.0f0
            if zrc_shifted > zvir_half
                zvir_half = zrc_shifted
            end

            jj_int += _I1
        end
    end

    @inbounds RTHL_out[peak_id]  = RTHL
    @inbounds Fbarx_out[peak_id] = Fbarx
    @inbounds e_v_out[peak_id]   = e_v_f
    @inbounds p_v_out[peak_id]   = p_v_f
    @inbounds Srb_out[peak_id]   = Srb
    @inbounds d2F_out[peak_id]   = d2F
    @inbounds zvir_half_out[peak_id] = zvir_half
    @inbounds Sbar_out[_I1, peak_id]  = Sbar_x
    @inbounds Sbar_out[Int32(2), peak_id] = Sbar_y
    @inbounds Sbar_out[Int32(3), peak_id] = Sbar_z
    @inbounds Sbar2_out[_I1, peak_id] = Sbar2_x
    @inbounds Sbar2_out[Int32(2), peak_id] = Sbar2_y
    @inbounds Sbar2_out[Int32(3), peak_id] = Sbar2_z
    @inbounds gradpk_out[_I1, peak_id]  = gx_mat
    @inbounds gradpk_out[Int32(2), peak_id] = gy_mat
    @inbounds gradpk_out[Int32(3), peak_id] = gz_mat
    @inbounds gradpkf_out[_I1, peak_id]  = gfx_mat
    @inbounds gradpkf_out[Int32(2), peak_id] = gfy_mat
    @inbounds gradpkf_out[Int32(3), peak_id] = gfz_mat
    @inbounds gradpkrf_out[_I1, peak_id]  = gradpkrf_x
    @inbounds gradpkrf_out[Int32(2), peak_id] = gradpkrf_y
    @inbounds gradpkrf_out[Int32(3), peak_id] = gradpkrf_z
    @inbounds strain_final_out[1, 1, peak_id] = E11f
    @inbounds strain_final_out[1, 2, peak_id] = E12f
    @inbounds strain_final_out[1, 3, peak_id] = E13f
    @inbounds strain_final_out[2, 1, peak_id] = E12f
    @inbounds strain_final_out[2, 2, peak_id] = E22f
    @inbounds strain_final_out[2, 3, peak_id] = E23f
    @inbounds strain_final_out[3, 1, peak_id] = E13f
    @inbounds strain_final_out[3, 2, peak_id] = E23f
    @inbounds strain_final_out[3, 3, peak_id] = E33f
    # eigenvalues of the trace-matched interpolated strain, not strain_final
    @inbounds eigs_out[1, peak_id] = lamf1
    @inbounds eigs_out[2, peak_id] = lamf2
    @inbounds eigs_out[3, peak_id] = lamf3

    # Mask update: set mask = 1 on in-bounds, non-buffer cells within RTHL.
    # Concurrent blocks may write the same cell; harmless since the value is always 1.
    if update_mask
        @inbounds ci = peaks_i[peak_id]
        @inbounds cj = peaks_j[peak_id]
        @inbounds ck = peaks_k[peak_id]
        s_mask = _I1
        while s_mask <= nshells
            @inbounds r2_s = shell_r2[s_mask]
            if sqrt(Float32(r2_s)) > RTHL
                break
            end
            @inbounds s0 = shell_start[s_mask]
            @inbounds nc = shell_count[s_mask]
            c_mask = _I1
            while c_mask <= nc
                idx = s0 + c_mask - _I1
                @inbounds di = off_di[idx]
                @inbounds dj = off_dj[idx]
                @inbounds dk = off_dk[idx]
                iv1 = ci + di; iv2 = cj + dj; iv3 = ck + dk
                if (_I1 <= iv1) & (iv1 <= n1) & (_I1 <= iv2) & (iv2 <= n2) & (_I1 <= iv3) & (iv3 <= n3)
                    if (iv1 > nbuff) & (iv1 <= n1 - nbuff) &
                       (iv2 > nbuff) & (iv2 <= n2 - nbuff) &
                       (iv3 > nbuff) & (iv3 <= n3 - nbuff)
                        @inbounds mask_out[iv1, iv2, iv3] = Int8(1)
                    end
                end
                c_mask += _I1
            end
            s_mask += _I1
        end
    end
    return
end

function Huntian.analyse_peak_gpu_cuda(
        delta_h, etax_h, etay_h, etaz_h, eta2x_h, eta2y_h, eta2z_h,
        peaks_i, peaks_j, peaks_k,
        stab::ShellTables,
        ct_table_h::AbstractArray{<:Real,3},
        X1::Real, X2::Real, Y1::Real, Y2::Real, Z1::Real, Z2::Real,
        alatt::Real, ir2min::Integer, ZZon::Real, Rfclvi::Real;
        fcrit_override::Union{Nothing, Real}=nothing,
        ct=nothing,
        rmax2rs::Real=0.0, threads::Int=128, ct_out_val::Real=-1.0,
        lapd::Union{Nothing, AbstractArray{<:Real,3}}=nothing,
        mask::Union{Nothing, AbstractArray{<:Integer,3}}=nothing,
        nbuff::Integer=0)
    # fcrit: explicit override, else from the collapse table, else 1.686
    fcrit = if fcrit_override !== nothing
        Float64(fcrit_override)
    elseif ct !== nothing
        Huntian.RadialShell.fsc_of_z(Float64(ZZon) - 1.0, ct)
    else
        1.686
    end
    npeaks = length(peaks_i)
    nshells = stab.nshells

    delta_d  = CuArray{Float32}(delta_h)
    etax_d   = CuArray{Float32}(etax_h)
    etay_d   = CuArray{Float32}(etay_h)
    etaz_d   = CuArray{Float32}(etaz_h)
    eta2x_d  = CuArray{Float32}(eta2x_h)
    eta2y_d  = CuArray{Float32}(eta2y_h)
    eta2z_d  = CuArray{Float32}(eta2z_h)
    lapd_d   = lapd === nothing ? CUDA.zeros(Float32, size(delta_h)...) :
                                   (@assert size(lapd) == size(delta_h); CuArray{Float32}(lapd))
    pi_d2 = CuArray{Int32}(peaks_i)
    pj_d2 = CuArray{Int32}(peaks_j)
    pk_d2 = CuArray{Int32}(peaks_k)
    stab_d = ShellTablesGPU(stab)

    Fshell_d    = CUDA.zeros(Float32, nshells, npeaks)
    nshell_d    = CUDA.zeros(Int32,   nshells, npeaks)
    Sshell_d    = CUDA.zeros(Float32, 3, nshells, npeaks)
    S2shell_d   = CUDA.zeros(Float32, 3, nshells, npeaks)
    Gshell_d    = CUDA.zeros(Float32, 3, nshells, npeaks)
    Gfshell_d   = CUDA.zeros(Float32, 3, nshells, npeaks)
    SRshell_d   = CUDA.zeros(Float32, 3, 3, nshells, npeaks)
    lapdshell_d = CUDA.zeros(Float32, nshells, npeaks)

    _launch_shell_gather_full!(
        Fshell_d, nshell_d, Sshell_d, S2shell_d, Gshell_d, Gfshell_d, SRshell_d,
        lapdshell_d,
        delta_d, etax_d, etay_d, etaz_d, eta2x_d, eta2y_d, eta2z_d, lapd_d,
        pi_d2, pj_d2, pk_d2, stab_d; threads=threads)

    # Field inputs are not needed after the gather; free them to make room
    # for the mask and outputs.
    CUDA.unsafe_free!(delta_d)
    CUDA.unsafe_free!(etax_d); CUDA.unsafe_free!(etay_d); CUDA.unsafe_free!(etaz_d)
    CUDA.unsafe_free!(eta2x_d); CUDA.unsafe_free!(eta2y_d); CUDA.unsafe_free!(eta2z_d)
    CUDA.unsafe_free!(lapd_d)

    shell_r2_d = CuArray{Int32}(stab.shell_r2)
    ct_table_d = CuArray{Float32}(ct_table_h)
    nx, ny, nz = size(ct_table_h)
    dxi = Float32((nx - 1) / (X2 - X1))
    dyi = Float32((ny - 1) / (Y2 - Y1))
    dzi = Float32((nz - 1) / (Z2 - Z1))

    # nshells - 1 matches RadialShell.analyse_peak, whose per-cell loop never
    # finalises the last shell. rmax2rs > 0 further limits r ≤ rmax2rs·Rf.
    nshells_max = nshells - 1
    if rmax2rs > 0.0
        r2_limit = (Float64(rmax2rs) * Float64(Rfclvi) / Float64(alatt))^2
        count = 0
        for s in 1:nshells - 1
            stab.shell_r2[s] <= r2_limit || break
            count = s
        end
        nshells_max = count
    end

    # Scaling constants as in CPU analyse_peak_gpu
    hlatt = 1.0
    anor = 1.0 / hlatt^3
    aRnor = 3.0 * anor / alatt
    wnor = anor / (4.0 * pi)
    wRnor = 3.0 * wnor / alatt
    hlatt_1 = 1.0
    hlatt_2 = 1.0

    RTHL_d          = CUDA.zeros(Float32, npeaks)
    Fbarx_d         = CUDA.zeros(Float32, npeaks)
    e_v_d           = CUDA.zeros(Float32, npeaks)
    p_v_d           = CUDA.zeros(Float32, npeaks)
    strain_final_d  = CUDA.zeros(Float32, 3, 3, npeaks)
    eigs_d          = CUDA.zeros(Float32, 3, npeaks)
    Srb_d           = CUDA.zeros(Float32, npeaks)
    Sbar_d          = CUDA.zeros(Float32, 3, npeaks)
    Sbar2_d         = CUDA.zeros(Float32, 3, npeaks)
    gradpk_d        = CUDA.zeros(Float32, 3, npeaks)
    gradpkf_d       = CUDA.zeros(Float32, 3, npeaks)
    gradpkrf_d      = CUDA.zeros(Float32, 3, npeaks)
    d2F_d           = CUDA.zeros(Float32, npeaks)
    zvir_half_d     = CUDA.zeros(Float32, npeaks)

    # Without a mask, pass a 1×1×1 dummy (never written; update_mask=false).
    update_mask = mask !== nothing
    n1_delta, n2_delta, n3_delta = size(delta_h)
    mask_d = if update_mask
        @assert size(mask) == size(delta_h)
        CuArray{Int8}(mask)
    else
        CuArray{Int8}(zeros(Int8, 1, 1, 1))
    end

    # Shared memory: rad + Fbar + 3*Gn + 3*Gfn + 9*SRn = 17 Float32 × _MAX_SHELLS_GPU
    _check_shell_capacity(stab.nshells)
    shmem_bytes = 17 * Int(_MAX_SHELLS_GPU) * sizeof(Float32)

    Rfclvi_r2 = Float32((Rfclvi / alatt)^2)

    # Single-call entry always broadcasts scalar ZZon/fcrit (per-peak values
    # are only supported via analyse_peaks_gpu_cuda_multirf).
    ZZon_pp_d  = CUDA.fill(Float32(ZZon),  npeaks)
    fcrit_pp_d = CUDA.fill(Float32(fcrit), npeaks)

    @cuda threads=threads blocks=npeaks shmem=shmem_bytes _post_process_kernel!(
        RTHL_d, Fbarx_d, e_v_d, p_v_d, strain_final_d, eigs_d,
        Srb_d, Sbar_d, Sbar2_d, gradpk_d, gradpkf_d, gradpkrf_d, d2F_d,
        zvir_half_d, mask_d,
        Fshell_d, nshell_d, Sshell_d, S2shell_d, Gshell_d, Gfshell_d, SRshell_d,
        lapdshell_d,
        pi_d2, pj_d2, pk_d2,
        stab_d.offsets_di, stab_d.offsets_dj, stab_d.offsets_dk,
        stab_d.shell_start, stab_d.shell_count,
        shell_r2_d,
        ct_table_d,
        Float32(X1), Float32(Y1), Float32(Z1),
        dxi, dyi, dzi,
        Int32(nx), Int32(ny), Int32(nz), Float32(ct_out_val),
        _AKK_GPU[],
        ZZon_pp_d, fcrit_pp_d, Rfclvi_r2,
        Float32(wRnor), Float32(aRnor), Float32(hlatt_1), Float32(hlatt_2),
        Int32(ir2min),
        Int32(nshells), Int32(nshells_max),
        Int32(n1_delta), Int32(n2_delta), Int32(n3_delta), Int32(nbuff), update_mask,
    )

    return (
        RTHL         = Array(RTHL_d),
        Fbarx        = Array(Fbarx_d),
        e_v          = Array(e_v_d),
        p_v          = Array(p_v_d),
        strain_final = Array(strain_final_d),
        eigs         = Array(eigs_d),
        Srb          = Array(Srb_d),
        Sbar         = Array(Sbar_d),
        Sbar2        = Array(Sbar2_d),
        gradpk       = Array(gradpk_d),
        gradpkf      = Array(gradpkf_d),
        gradpkrf     = Array(gradpkrf_d),
        d2F          = Array(d2F_d),
        zvir_half    = Array(zvir_half_d),
        mask         = update_mask ? Array(mask_d) : nothing,
    )
end

# Multi-Rf variant: uploads the fields, lapd and mask once, then runs gather +
# post-process for each entry of `batches`, avoiding a re-upload per filter scale.
# Batch entries: `(peaks_i, peaks_j, peaks_k, Rf, ir2min)` for ievol==0, or with
# trailing per-peak `(ZZon_pp, fcrit_pp)` arrays for ievol==1.

function Huntian.analyse_peaks_gpu_cuda_multirf(
        delta_h, etax_h, etay_h, etaz_h, eta2x_h, eta2y_h, eta2z_h,
        batches::AbstractVector,
        stab::ShellTables,
        ct_table_h::AbstractArray{<:Real,3},
        X1::Real, X2::Real, Y1::Real, Y2::Real, Z1::Real, Z2::Real,
        alatt::Real, ZZon::Real;
        fcrit_override::Union{Nothing, Real}=nothing,
        ct=nothing,
        rmax2rs::Real=0.0, threads::Int=128, ct_out_val::Real=-1.0,
        lapd::Union{Nothing, AbstractArray{<:Real,3}}=nothing,
        mask::Union{Nothing, AbstractArray{<:Integer,3}}=nothing,
        nbuff::Integer=0)

    fcrit = if fcrit_override !== nothing
        Float64(fcrit_override)
    elseif ct !== nothing
        Huntian.RadialShell.fsc_of_z(Float64(ZZon) - 1.0, ct)
    else
        1.686
    end
    nshells = stab.nshells

    # Inputs already on the device are used directly and stay caller-owned;
    # only arrays uploaded here are freed at the end.
    _upload(x) = x isa CuArray{Float32,3} ? (x, false) : (CuArray{Float32}(x), true)
    delta_d,  _free_delta  = _upload(delta_h)
    etax_d,   _free_etax   = _upload(etax_h)
    etay_d,   _free_etay   = _upload(etay_h)
    etaz_d,   _free_etaz   = _upload(etaz_h)
    eta2x_d,  _free_eta2x  = _upload(eta2x_h)
    eta2y_d,  _free_eta2y  = _upload(eta2y_h)
    eta2z_d,  _free_eta2z  = _upload(eta2z_h)
    _free_lapd = false
    if lapd === nothing
        lapd_d = CUDA.zeros(Float32, size(delta_d)...)
        _free_lapd = true
    elseif lapd isa CuArray{Float32,3}
        @assert size(lapd) == size(delta_d)
        lapd_d = lapd
    else
        @assert size(lapd) == size(delta_d)
        lapd_d = CuArray{Float32}(lapd)
        _free_lapd = true
    end
    stab_d     = ShellTablesGPU(stab)
    shell_r2_d = CuArray{Int32}(stab.shell_r2)
    ct_table_d = CuArray{Float32}(ct_table_h)

    update_mask = mask !== nothing
    n1_delta, n2_delta, n3_delta = size(delta_d)
    mask_d = if update_mask
        @assert size(mask) == size(delta_d)
        CuArray{Int8}(mask)
    else
        CuArray{Int8}(zeros(Int8, 1, 1, 1))
    end

    nx, ny, nz = size(ct_table_h)
    dxi = Float32((nx - 1) / (X2 - X1))
    dyi = Float32((ny - 1) / (Y2 - Y1))
    dzi = Float32((nz - 1) / (Z2 - Z1))

    hlatt = 1.0
    anor = 1.0 / hlatt^3
    aRnor = 3.0 * anor / alatt
    wnor = anor / (4.0 * pi)
    wRnor = 3.0 * wnor / alatt
    hlatt_1 = 1.0
    hlatt_2 = 1.0

    _check_shell_capacity(stab.nshells)
    shmem_bytes = 17 * Int(_MAX_SHELLS_GPU) * sizeof(Float32)

    # Chunk peaks so per-peak shell profiles (24 Float32 × nshells) use at
    # most ~8 GB, leaving room for the fields.
    _shell_bytes_per_peak = 24 * sizeof(Float32) * nshells
    _gpu_shell_budget = 8 * 1024^3
    _max_peaks_per_chunk = max(1024, _gpu_shell_budget ÷ _shell_bytes_per_peak)

    results = Vector{NamedTuple}(undef, length(batches))

    for (bi, batch) in enumerate(batches)
        peaks_i = batch[1]; peaks_j = batch[2]; peaks_k = batch[3]
        Rf      = batch[4]; ir2min  = batch[5]
        has_pp_b = length(batch) >= 7
        ZZon_pp_b_h  = has_pp_b ? batch[6] : nothing
        fcrit_pp_b_h = has_pp_b ? batch[7] : nothing
        npeaks_total = length(peaks_i)

        # Excludes the last shell, as on CPU (see analyse_peak_gpu_cuda).
        nshells_max = nshells - 1
        if rmax2rs > 0.0
            r2_limit = (Float64(rmax2rs) * Float64(Rf) / Float64(alatt))^2
            count = 0
            for s in 1:nshells - 1
                stab.shell_r2[s] <= r2_limit || break
                count = s
            end
            nshells_max = count
        end

        Rfclvi_r2 = Float32((Rf / alatt)^2)

        nchunks = cld(npeaks_total, _max_peaks_per_chunk)

        RTHL_h          = Vector{Float32}(undef, npeaks_total)
        Fbarx_h         = Vector{Float32}(undef, npeaks_total)
        e_v_h           = Vector{Float32}(undef, npeaks_total)
        p_v_h           = Vector{Float32}(undef, npeaks_total)
        strain_final_h  = Array{Float32}(undef, 3, 3, npeaks_total)
        eigs_h          = Array{Float32}(undef, 3, npeaks_total)
        Srb_h           = Vector{Float32}(undef, npeaks_total)
        Sbar_h          = Array{Float32}(undef, 3, npeaks_total)
        Sbar2_h         = Array{Float32}(undef, 3, npeaks_total)
        gradpk_h        = Array{Float32}(undef, 3, npeaks_total)
        gradpkf_h       = Array{Float32}(undef, 3, npeaks_total)
        gradpkrf_h      = Array{Float32}(undef, 3, npeaks_total)
        d2F_h           = Vector{Float32}(undef, npeaks_total)
        zvir_half_h     = Vector{Float32}(undef, npeaks_total)

        for chunk in 1:nchunks
            c_start = (chunk - 1) * _max_peaks_per_chunk + 1
            c_end   = min(chunk * _max_peaks_per_chunk, npeaks_total)
            npeaks  = c_end - c_start + 1
            rng     = c_start:c_end

            pi_d = CuArray{Int32}(view(peaks_i, rng))
            pj_d = CuArray{Int32}(view(peaks_j, rng))
            pk_d = CuArray{Int32}(view(peaks_k, rng))

            Fshell_d    = CUDA.zeros(Float32, nshells, npeaks)
            nshell_d    = CUDA.zeros(Int32,   nshells, npeaks)
            Sshell_d    = CUDA.zeros(Float32, 3, nshells, npeaks)
            S2shell_d   = CUDA.zeros(Float32, 3, nshells, npeaks)
            Gshell_d    = CUDA.zeros(Float32, 3, nshells, npeaks)
            Gfshell_d   = CUDA.zeros(Float32, 3, nshells, npeaks)
            SRshell_d   = CUDA.zeros(Float32, 3, 3, nshells, npeaks)
            lapdshell_d = CUDA.zeros(Float32, nshells, npeaks)

            ZZon_pp_d  = has_pp_b ? CuArray{Float32}(view(ZZon_pp_b_h, rng))  :
                                    CUDA.fill(Float32(ZZon),  npeaks)
            fcrit_pp_d = has_pp_b ? CuArray{Float32}(view(fcrit_pp_b_h, rng)) :
                                    CUDA.fill(Float32(fcrit), npeaks)

            _launch_shell_gather_full!(
                Fshell_d, nshell_d, Sshell_d, S2shell_d, Gshell_d, Gfshell_d, SRshell_d,
                lapdshell_d,
                delta_d, etax_d, etay_d, etaz_d, eta2x_d, eta2y_d, eta2z_d, lapd_d,
                pi_d, pj_d, pk_d, stab_d; threads=threads,
                # early exit is exact only with the full shell range (rmax2rs == 0)
                shell_r2_d=(rmax2rs > 0.0 ? nothing : shell_r2_d), ir2min=ir2min,
                fcrit_pp_d=fcrit_pp_d)

            RTHL_d          = CUDA.zeros(Float32, npeaks)
            Fbarx_d         = CUDA.zeros(Float32, npeaks)
            e_v_d           = CUDA.zeros(Float32, npeaks)
            p_v_d           = CUDA.zeros(Float32, npeaks)
            strain_final_d  = CUDA.zeros(Float32, 3, 3, npeaks)
            eigs_d          = CUDA.zeros(Float32, 3, npeaks)
            Srb_d           = CUDA.zeros(Float32, npeaks)
            Sbar_d          = CUDA.zeros(Float32, 3, npeaks)
            Sbar2_d         = CUDA.zeros(Float32, 3, npeaks)
            gradpk_d        = CUDA.zeros(Float32, 3, npeaks)
            gradpkf_d       = CUDA.zeros(Float32, 3, npeaks)
            gradpkrf_d      = CUDA.zeros(Float32, 3, npeaks)
            d2F_d           = CUDA.zeros(Float32, npeaks)
            zvir_half_d     = CUDA.zeros(Float32, npeaks)

            @cuda threads=threads blocks=npeaks shmem=shmem_bytes _post_process_kernel!(
                RTHL_d, Fbarx_d, e_v_d, p_v_d, strain_final_d, eigs_d,
                Srb_d, Sbar_d, Sbar2_d, gradpk_d, gradpkf_d, gradpkrf_d, d2F_d,
                zvir_half_d, mask_d,
                Fshell_d, nshell_d, Sshell_d, S2shell_d, Gshell_d, Gfshell_d, SRshell_d,
                lapdshell_d,
                pi_d, pj_d, pk_d,
                stab_d.offsets_di, stab_d.offsets_dj, stab_d.offsets_dk,
                stab_d.shell_start, stab_d.shell_count,
                shell_r2_d,
                ct_table_d,
                Float32(X1), Float32(Y1), Float32(Z1),
                dxi, dyi, dzi,
                Int32(nx), Int32(ny), Int32(nz), Float32(ct_out_val),
                _AKK_GPU[],
                ZZon_pp_d, fcrit_pp_d, Rfclvi_r2,
                Float32(wRnor), Float32(aRnor), Float32(hlatt_1), Float32(hlatt_2),
                Int32(ir2min),
                Int32(nshells), Int32(nshells_max),
                Int32(n1_delta), Int32(n2_delta), Int32(n3_delta), Int32(nbuff), update_mask,
            )

            copyto!(RTHL_h, c_start, Array(RTHL_d), 1, npeaks)
            copyto!(Fbarx_h, c_start, Array(Fbarx_d), 1, npeaks)
            copyto!(e_v_h, c_start, Array(e_v_d), 1, npeaks)
            copyto!(p_v_h, c_start, Array(p_v_d), 1, npeaks)
            copyto!(Srb_h, c_start, Array(Srb_d), 1, npeaks)
            copyto!(d2F_h, c_start, Array(d2F_d), 1, npeaks)
            copyto!(zvir_half_h, c_start, Array(zvir_half_d), 1, npeaks)

            strain_chunk = Array(strain_final_d)
            eigs_chunk   = Array(eigs_d)
            Sbar_chunk   = Array(Sbar_d)
            Sbar2_chunk  = Array(Sbar2_d)
            gradpk_chunk = Array(gradpk_d)
            gradpkf_chunk = Array(gradpkf_d)
            gradpkrf_chunk = Array(gradpkrf_d)
            strain_final_h[:, :, rng] .= strain_chunk
            eigs_h[:, rng]            .= eigs_chunk
            Sbar_h[:, rng]            .= Sbar_chunk
            Sbar2_h[:, rng]           .= Sbar2_chunk
            gradpk_h[:, rng]          .= gradpk_chunk
            gradpkf_h[:, rng]         .= gradpkf_chunk
            gradpkrf_h[:, rng]        .= gradpkrf_chunk

            CUDA.unsafe_free!(Fshell_d); CUDA.unsafe_free!(nshell_d)
            CUDA.unsafe_free!(Sshell_d); CUDA.unsafe_free!(S2shell_d)
            CUDA.unsafe_free!(Gshell_d); CUDA.unsafe_free!(Gfshell_d)
            CUDA.unsafe_free!(SRshell_d); CUDA.unsafe_free!(lapdshell_d)
            CUDA.unsafe_free!(pi_d); CUDA.unsafe_free!(pj_d); CUDA.unsafe_free!(pk_d)
            CUDA.unsafe_free!(RTHL_d); CUDA.unsafe_free!(Fbarx_d)
            CUDA.unsafe_free!(e_v_d); CUDA.unsafe_free!(p_v_d)
            CUDA.unsafe_free!(strain_final_d); CUDA.unsafe_free!(eigs_d)
            CUDA.unsafe_free!(Srb_d); CUDA.unsafe_free!(Sbar_d); CUDA.unsafe_free!(Sbar2_d)
            CUDA.unsafe_free!(gradpk_d); CUDA.unsafe_free!(gradpkf_d); CUDA.unsafe_free!(gradpkrf_d)
            CUDA.unsafe_free!(d2F_d); CUDA.unsafe_free!(zvir_half_d)
        end

        results[bi] = (
            RTHL         = RTHL_h,
            Fbarx        = Fbarx_h,
            e_v          = e_v_h,
            p_v          = p_v_h,
            strain_final = strain_final_h,
            eigs         = eigs_h,
            Srb          = Srb_h,
            Sbar         = Sbar_h,
            Sbar2        = Sbar2_h,
            gradpk       = gradpk_h,
            gradpkf      = gradpkf_h,
            gradpkrf     = gradpkrf_h,
            d2F          = d2F_h,
            zvir_half    = zvir_half_h,
        )
    end

    # Free only what was uploaded here.
    _free_delta && CUDA.unsafe_free!(delta_d)
    _free_etax  && CUDA.unsafe_free!(etax_d)
    _free_etay  && CUDA.unsafe_free!(etay_d)
    _free_etaz  && CUDA.unsafe_free!(etaz_d)
    _free_eta2x && CUDA.unsafe_free!(eta2x_d)
    _free_eta2y && CUDA.unsafe_free!(eta2y_d)
    _free_eta2z && CUDA.unsafe_free!(eta2z_d)
    _free_lapd  && CUDA.unsafe_free!(lapd_d)
    CUDA.unsafe_free!(shell_r2_d); CUDA.unsafe_free!(ct_table_d)

    mask_out = update_mask ? Array(mask_d) : nothing
    CUDA.unsafe_free!(mask_d)
    return (results=results, mask=mask_out)
end

# Isolated (zero-padded) FFT convolution, GPU version of
# `MultiResolution._isolated_convolve`:
#   noise (n³) → zero-pad to (2n)³ → rfft → × √P(k) · transfer(k) → irfft → extract tile.
# √P(k) is tabulated in log k on the host.

@inline function _interp_pk_table(pk_table, log_k::Float32, log_k_min::Float32,
                                  d_log_k::Float32, ntable::Int32)
    fpos = (log_k - log_k_min) / d_log_k
    i0 = Int32(floor(fpos)) + _I1
    if i0 < _I1
        @inbounds return pk_table[1]
    elseif i0 >= ntable
        @inbounds return pk_table[ntable]
    else
        frac = fpos - Float32(i0 - _I1)
        @inbounds return pk_table[i0] + frac * (pk_table[i0 + _I1] - pk_table[i0])
    end
end

# One thread per mode of the rfft buffer `padded_k` (n2÷2+1, n2, n2).
# kernel_fn_id: 0 δ, 1 1LPT, 2 2LPT, 3 φ_ij, 4 Laplacian, 5 potential.
function _transfer_kernel!(padded_k, pk_table, log_k_min::Float32,
                            d_log_k::Float32, ntable::Int32,
                            dk::Float64, n2::Int32,
                            kernel_fn_id::Int32, dim1::Int32, dim2::Int32)
    nk = (n2 >> _I1) + _I1
    total = Int64(nk) * Int64(n2) * Int64(n2)
    idx0 = Int64(blockIdx().x - _I1) * Int64(blockDim().x) + Int64(threadIdx().x)
    if idx0 > total
        return
    end
    li = idx0 - Int64(1)
    ix = Int32(li % Int64(nk)) + _I1
    rem = li ÷ Int64(nk)
    iy = Int32(rem % Int64(n2)) + _I1
    iz = Int32(rem ÷ Int64(n2)) + _I1

    # Float64 frequencies as on CPU; Float32 gives ~1e-5 per-mode error at n≥64.
    half = (n2 >> _I1)
    kx = Float64(ix - _I1) * dk
    iy_signed = iy <= half ? (iy - _I1) : (iy - n2 - _I1)
    iz_signed = iz <= half ? (iz - _I1) : (iz - n2 - _I1)
    ky = Float64(iy_signed) * dk
    kz = Float64(iz_signed) * dk
    k2 = kx * kx + ky * ky + kz * kz

    if k2 == 0.0
        @inbounds padded_k[ix, iy, iz] = ComplexF32(0)
        return
    end
    sqrt_pk = Float64(_interp_pk_table(pk_table, Float32(log(k2) * 0.5),
                                        log_k_min, d_log_k, ntable))

    @inbounds val = padded_k[ix, iy, iz]

    if kernel_fn_id == _I0
        # δ: √P(k)
        @inbounds padded_k[ix, iy, iz] = ComplexF32(val * sqrt_pk)
    elseif kernel_fn_id == _I1
        # 1LPT: i * ki / k² × √P(k)
        ki = (dim1 == _I1) ? kx : ((dim1 == Int32(2)) ? ky : kz)
        coeff = sqrt_pk * ki / k2
        a = Float64(real(val)); b = Float64(imag(val))
        @inbounds padded_k[ix, iy, iz] = ComplexF32(-b * coeff, a * coeff)
    elseif kernel_fn_id == Int32(2)
        # 2LPT: -i * ki / k² × √P(k)
        ki = (dim1 == _I1) ? kx : ((dim1 == Int32(2)) ? ky : kz)
        coeff = sqrt_pk * ki / k2
        a = Float64(real(val)); b = Float64(imag(val))
        @inbounds padded_k[ix, iy, iz] = ComplexF32(b * coeff, -a * coeff)
    elseif kernel_fn_id == Int32(3)
        # φ_ij: -ki*kj/k² × √P(k)  (real coefficient)
        ki = (dim1 == _I1) ? kx : ((dim1 == Int32(2)) ? ky : kz)
        kj = (dim2 == _I1) ? kx : ((dim2 == Int32(2)) ? ky : kz)
        coeff = Float32(-sqrt_pk * ki * kj / k2)
        @inbounds padded_k[ix, iy, iz] = val * coeff
    elseif kernel_fn_id == Int32(4)
        # Laplacian: k² × √P(k)
        coeff = Float32(sqrt_pk * k2)
        @inbounds padded_k[ix, iy, iz] = val * coeff
    elseif kernel_fn_id == Int32(5)
        # ∇⁻² (potential): -1/k² × √P(k)  (real coefficient)
        coeff = Float32(-sqrt_pk / k2)
        @inbounds padded_k[ix, iy, iz] = val * coeff
    end
    return
end

# Enforce Hermitian symmetry on the DC (ix=1) and Nyquist (ix=n2/2+1) planes
# of an rfft buffer. On Hermitian-violating input (e.g. off-diagonal φ_ij),
# cuFFT's C2R differs from FFTW's irfft by up to ~5%; FFTW effectively averages
# conjugate pairs, so we do that explicitly: F[iy,iz] and F[iy_c,iz_c] are
# replaced by their Hermitian average, and self-conjugate cells made real.
function _hermitize_planes_kernel!(padded_k, n2::Int32, plane_ix1::Int32,
                                    plane_ix2::Int32)
    idx0 = (blockIdx().x - _I1) * blockDim().x + threadIdx().x
    total = Int64(n2) * Int64(n2)
    if idx0 > total
        return
    end
    li = idx0 - _I1
    iy = Int32(li % Int64(n2)) + _I1
    iz = Int32(li ÷ Int64(n2)) + _I1
    iy_c = iy == _I1 ? _I1 : (n2 - iy + Int32(2))
    iz_c = iz == _I1 ? _I1 : (n2 - iz + Int32(2))

    # Avoid races: only the lexicographically lower cell of each conjugate
    # pair works, writing both slots.
    is_lower = (iy < iy_c) | ((iy == iy_c) & (iz < iz_c))
    is_self  = (iy == iy_c) & (iz == iz_c)
    if !(is_lower | is_self)
        return
    end

    for ix in (plane_ix1, plane_ix2)
        if is_self
            @inbounds val = padded_k[ix, iy, iz]
            @inbounds padded_k[ix, iy, iz] = ComplexF32(real(val), 0.0f0)
        else
            @inbounds vme = padded_k[ix, iy, iz]
            @inbounds vco = padded_k[ix, iy_c, iz_c]
            avg = (vme + conj(vco)) * 0.5f0
            @inbounds padded_k[ix, iy, iz]     = avg
            @inbounds padded_k[ix, iy_c, iz_c] = conj(avg)
        end
    end
    return
end

function Huntian.isolated_convolve_gpu(noise::AbstractArray{<:Real,3},
                                          pk, boxsize_local::Real, n::Integer;
                                          kernel_fn_id::Integer=0,
                                          dim1::Integer=0, dim2::Integer=0,
                                          nshell::Integer=0,
                                          pk_table_n::Integer=8192,
                                          threads::Integer=256)
    n_input = size(noise, 1)
    @assert size(noise) == (n_input, n_input, n_input)
    n2 = 2 * n_input
    dx = Float64(boxsize_local) / n
    dk = 2π / (n2 * dx)

    # Table of √(P(k) · dk³ · n2³) in log k, covering dk/2 .. 1.5·√3·k_Nyquist.
    k_min_raw = dk
    k_max_raw = sqrt(3.0) * (n2 / 2) * dk
    log_k = range(log(k_min_raw * 0.5), log(k_max_raw * 1.5), length=pk_table_n)
    norm = dk^3 * Float64(n2)^3
    pk_vals = Float32[sqrt(Float64(pk(exp(lk))) * norm) for lk in log_k]
    pk_table_d = CuArray(pk_vals)
    log_k_min_f = Float32(log_k[1])
    d_log_k_f   = Float32(step(log_k))

    padded_d = CUDA.zeros(Float32, n2, n2, n2)
    noise_d = CuArray{Float32}(noise)
    @inbounds padded_d[1:n_input, 1:n_input, 1:n_input] .= noise_d
    CUDA.unsafe_free!(noise_d)

    padded_k_d = _cufft_rfft(padded_d)
    CUDA.unsafe_free!(padded_d)

    nk = size(padded_k_d, 1)
    total = nk * n2 * n2
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _transfer_kernel!(
        padded_k_d, pk_table_d,
        log_k_min_f, d_log_k_f, Int32(pk_table_n),
        Float64(dk), Int32(n2),
        Int32(kernel_fn_id), Int32(dim1), Int32(dim2),
    )

    # Required for cuFFT irfft to match FFTW (see _hermitize_planes_kernel!).
    plane_total = n2 * n2
    plane_blocks = cld(plane_total, threads)
    @cuda threads=threads blocks=plane_blocks _hermitize_planes_kernel!(
        padded_k_d, Int32(n2), _I1, Int32(n2 ÷ 2 + 1))

    result_d = _cufft_irfft(padded_k_d, n2)
    CUDA.unsafe_free!(padded_k_d)

    s1 = nshell + 1
    s2 = nshell + n
    tile_d = result_d[s1:s2, s1:s2, s1:s2]
    CUDA.unsafe_free!(result_d)

    return Array(tile_d)
end

# Isolated convolution with several output kernels sharing one forward rFFT
# of the noise. `kernels` holds (kernel_fn_id, dim1, dim2) tuples as in
# isolated_convolve_gpu. Returns one (n,n,n) Float32 array per kernel (on
# device if return_device). Hermitisation is per kernel, since transfer
# kernels such as 1LPT break symmetry on the DC/Nyquist planes.
function Huntian.isolated_convolve_gpu_multi(noise::AbstractArray{<:Real,3},
                                                pk, boxsize_local::Real, n::Integer;
                                                kernels::AbstractVector,
                                                nshell::Integer=0,
                                                pk_table_n::Integer=8192,
                                                threads::Integer=256,
                                                return_device::Bool=false)
    n_input = size(noise, 1)
    @assert size(noise) == (n_input, n_input, n_input)
    n2 = 2 * n_input
    dx = Float64(boxsize_local) / n
    dk = 2π / (n2 * dx)

    k_min_raw = dk
    k_max_raw = sqrt(3.0) * (n2 / 2) * dk
    log_k = range(log(k_min_raw * 0.5), log(k_max_raw * 1.5), length=pk_table_n)
    norm = dk^3 * Float64(n2)^3
    pk_vals = Float32[sqrt(Float64(pk(exp(lk))) * norm) for lk in log_k]
    pk_table_d = CuArray(pk_vals)
    log_k_min_f = Float32(log_k[1])
    d_log_k_f   = Float32(step(log_k))

    padded_d = CUDA.zeros(Float32, n2, n2, n2)
    noise_d  = CuArray{Float32}(noise)
    @inbounds padded_d[1:n_input, 1:n_input, 1:n_input] .= noise_d
    CUDA.unsafe_free!(noise_d)
    base_k_d = _cufft_rfft(padded_d)
    CUDA.unsafe_free!(padded_d)

    nk = size(base_k_d, 1)
    total = nk * n2 * n2
    blocks = cld(total, threads)
    plane_total = n2 * n2
    plane_blocks = cld(plane_total, threads)
    s1 = nshell + 1
    s2 = nshell + n

    if return_device
        out_dev = Vector{CuArray{Float32,3}}(undef, length(kernels))
        for (ki, ktup) in enumerate(kernels)
            kernel_fn_id, dim1, dim2 = ktup
            scratch_k_d = copy(base_k_d)
            @cuda threads=threads blocks=blocks _transfer_kernel!(
                scratch_k_d, pk_table_d,
                log_k_min_f, d_log_k_f, Int32(pk_table_n),
                Float64(dk), Int32(n2),
                Int32(kernel_fn_id), Int32(dim1), Int32(dim2),
            )
            @cuda threads=threads blocks=plane_blocks _hermitize_planes_kernel!(
                scratch_k_d, Int32(n2), _I1, Int32(n2 ÷ 2 + 1))
            result_d = _cufft_irfft(scratch_k_d, n2)
            CUDA.unsafe_free!(scratch_k_d)
            # Copy into an owning CuArray so the caller can free it independently of result_d.
            out_dev[ki] = CuArray{Float32}(result_d[s1:s2, s1:s2, s1:s2])
            CUDA.unsafe_free!(result_d)
        end
        CUDA.unsafe_free!(base_k_d)
        CUDA.unsafe_free!(pk_table_d)
        return out_dev
    else
        out = Vector{Array{Float32,3}}(undef, length(kernels))
        for (ki, ktup) in enumerate(kernels)
            kernel_fn_id, dim1, dim2 = ktup
            scratch_k_d = copy(base_k_d)
            @cuda threads=threads blocks=blocks _transfer_kernel!(
                scratch_k_d, pk_table_d,
                log_k_min_f, d_log_k_f, Int32(pk_table_n),
                Float64(dk), Int32(n2),
                Int32(kernel_fn_id), Int32(dim1), Int32(dim2),
            )
            @cuda threads=threads blocks=plane_blocks _hermitize_planes_kernel!(
                scratch_k_d, Int32(n2), _I1, Int32(n2 ÷ 2 + 1))
            result_d = _cufft_irfft(scratch_k_d, n2)
            CUDA.unsafe_free!(scratch_k_d)
            tile_d = result_d[s1:s2, s1:s2, s1:s2]
            CUDA.unsafe_free!(result_d)
            out[ki] = Array(tile_d)
            CUDA.unsafe_free!(tile_d)
        end
        CUDA.unsafe_free!(base_k_d)
        CUDA.unsafe_free!(pk_table_d)
        return out
    end
end

# Periodic k-space kernel on an rfft buffer (as `MultiResolution._apply_kernel_inplace!`;
# no P(k)). Zeros k=0 and the Nyquist planes (ix=nk, iy=nyq+1, iz=nyq+1).
# kernel_fn_id:
#   2 = -i * k_{dim1} / k²         (2LPT displacement)
#   3 = -k_{dim1} * k_{dim2} / k²  (φ_ij)
#   4 = k²                         (Laplacian)
function _periodic_kernel!(arr_k, dk::Float64, n::Int32,
                            kernel_fn_id::Int32, dim1::Int32, dim2::Int32)
    nk = (n >> _I1) + _I1
    nyq = n >> _I1
    total = Int64(nk) * Int64(n) * Int64(n)
    idx0 = Int64(blockIdx().x - _I1) * Int64(blockDim().x) + Int64(threadIdx().x)
    if idx0 > total
        return
    end
    li = idx0 - Int64(1)
    ix = Int32(li % Int64(nk)) + _I1
    rem = li ÷ Int64(nk)
    iy = Int32(rem % Int64(n)) + _I1
    iz = Int32(rem ÷ Int64(n)) + _I1

    iy_signed = iy <= (nyq + _I1) ? (iy - _I1) : (iy - n - _I1)
    iz_signed = iz <= (nyq + _I1) ? (iz - _I1) : (iz - n - _I1)
    kx = Float64(ix - _I1) * dk
    ky = Float64(iy_signed) * dk
    kz = Float64(iz_signed) * dk
    k2 = kx * kx + ky * ky + kz * kz

    # CPU zeroing: DC origin, Nyquist rows/plane — except φ_ij (id 3), which zeroes only
    # k=0 so the 2LPT trace identity src2 = δ²/2 − Σφii²/2 − Σφij² holds with the
    # un-zeroed δ (serial convention).
    # −ki·kj/k² is real, so the Nyquist planes stay real.
    if k2 == 0.0 || (kernel_fn_id != Int32(3) &&
                     (ix == nk || iy == nyq + _I1 || iz == nyq + _I1))
        @inbounds arr_k[ix, iy, iz] = ComplexF32(0)
        return
    end

    @inbounds val = arr_k[ix, iy, iz]
    if kernel_fn_id == Int32(2)
        ki = (dim1 == _I1) ? kx : ((dim1 == Int32(2)) ? ky : kz)
        coeff = ki / k2
        a = Float64(real(val)); b = Float64(imag(val))
        @inbounds arr_k[ix, iy, iz] = ComplexF32(b * coeff, -a * coeff)
    elseif kernel_fn_id == Int32(3)
        ki = (dim1 == _I1) ? kx : ((dim1 == Int32(2)) ? ky : kz)
        kj = (dim2 == _I1) ? kx : ((dim2 == Int32(2)) ? ky : kz)
        coeff = Float32(-ki * kj / k2)
        @inbounds arr_k[ix, iy, iz] = val * coeff
    elseif kernel_fn_id == Int32(4)
        coeff = Float32(k2)
        @inbounds arr_k[ix, iy, iz] = val * coeff
    end
    return
end

function Huntian.isolated_poisson_psi_gpu(dS::CuArray{Float32,3}, boxsize_local::Real;
                                           threads::Integer=256)
    n = size(dS, 1); n2 = 2n
    padded = CUDA.zeros(Float32, n2, n2, n2)
    padded[1:n, 1:n, 1:n] .= dS
    fk = _cufft_rfft(padded)
    CUDA.unsafe_free!(padded)
    dk = 2π / (2 * Float64(boxsize_local))          # padded box = 2 × tile box
    nk = size(fk, 1)
    blocks = cld(nk * n2 * n2, threads)
    out = Any[]
    for dim in 1:3
        g = copy(fk)
        # kernel 2 is −i k_dim/k²; the 1LPT displacement is its negative
        @cuda threads=threads blocks=blocks _periodic_kernel!(g, Float64(dk), Int32(n2), Int32(2), Int32(dim), Int32(0))
        r = _cufft_irfft(g, n2)
        CUDA.unsafe_free!(g)
        push!(out, -r[1:n, 1:n, 1:n])
        CUDA.unsafe_free!(r)
    end
    CUDA.unsafe_free!(fk)
    return Tuple(out)
end

function Huntian.isolated_short_pot_gpu(delta::CuArray{Float32,3}, boxsize_local::Real, rs::Real)
    n = size(delta, 1); n2 = 2n
    padded = CUDA.zeros(Float32, n2, n2, n2)
    padded[1:n, 1:n, 1:n] .= delta
    fk = _cufft_rfft(padded)
    CUDA.unsafe_free!(padded)
    dk = 2π / (2 * Float64(boxsize_local)); nk = size(fk, 1)
    kx = CuArray(reshape(collect(0:nk-1) .* dk, nk, 1, 1))
    fq = [(i <= n2 ÷ 2 ? i : i - n2) * dk for i in 0:n2-1]
    ky = CuArray(reshape(fq, 1, n2, 1)); kz = CuArray(reshape(fq, 1, 1, n2))
    rs2 = Float64(rs)^2
    # −(1 − e^{−k²r_s²})/k², → −r_s² at k = 0 (CPU _isolated_short_pot parity)
    fk .*= ComplexF32.(_short_pot_kernel.(kx .^ 2 .+ ky .^ 2 .+ kz .^ 2, rs2))
    r = _cufft_irfft(fk, n2)
    CUDA.unsafe_free!(fk)
    out = r[1:n, 1:n, 1:n]
    CUDA.unsafe_free!(r)
    return out
end
@inline _short_pot_kernel(k2::Float64, rs2::Float64) = k2 == 0.0 ? -rs2 : -(1 - exp(-k2 * rs2)) / k2

function Huntian.compute_2lpt_gpu(delta_tile::AbstractArray{<:Real,3},
                                      nmesh::Integer, boxsize_local::Real;
                                      threads::Integer=256,
                                      return_device::Bool=false)
    n = Int(nmesh)
    @assert size(delta_tile) == (n, n, n)
    dk = 2π / Float64(boxsize_local)

    # A CuArray input is used in place and left to the caller to free.
    delta_d_is_input = delta_tile isa CuArray{Float32,3}
    delta_d = delta_d_is_input ? delta_tile : CuArray{Float32}(delta_tile)
    delta_k_d = _cufft_rfft(delta_d)

    # src2 = 0.5 * δ²
    src2_d = delta_d .^ 2 .* 0.5f0
    if !delta_d_is_input
        CUDA.unsafe_free!(delta_d)
    end

    nk = size(delta_k_d, 1)
    total = nk * n * n
    blocks = cld(total, threads)

    # src2 = δ²/2 − Σ_i φ_ii²/2 − Σ_{i<j} φ_ij²
    for (di, dj, coef) in ((1, 1, 0.5f0), (2, 2, 0.5f0), (3, 3, 0.5f0),
                            (1, 2, 1.0f0), (1, 3, 1.0f0), (2, 3, 1.0f0))
        phi_k_d = copy(delta_k_d)
        @cuda threads=threads blocks=blocks _periodic_kernel!(
            phi_k_d, Float64(dk), Int32(n),
            Int32(3), Int32(di), Int32(dj))
        phi_r_d = _cufft_irfft(phi_k_d, n)
        CUDA.unsafe_free!(phi_k_d)
        src2_d .-= phi_r_d .^ 2 .* coef
        CUDA.unsafe_free!(phi_r_d)
    end
    CUDA.unsafe_free!(delta_k_d)

    src2_k_d = _cufft_rfft(src2_d)
    CUDA.unsafe_free!(src2_d)

    if return_device
        psi2_dev = Vector{CuArray{Float32,3}}(undef, 3)
        for dim in 1:3
            psi2_k_d = copy(src2_k_d)
            @cuda threads=threads blocks=blocks _periodic_kernel!(
                psi2_k_d, Float64(dk), Int32(n),
                Int32(2), Int32(dim), Int32(0))
            psi2_dev[dim] = _cufft_irfft(psi2_k_d, n)
            CUDA.unsafe_free!(psi2_k_d)
        end
        CUDA.unsafe_free!(src2_k_d)
        return psi2_dev
    else
        psi2_host = Vector{Array{Float32,3}}(undef, 3)
        for dim in 1:3
            psi2_k_d = copy(src2_k_d)
            @cuda threads=threads blocks=blocks _periodic_kernel!(
                psi2_k_d, Float64(dk), Int32(n),
                Int32(2), Int32(dim), Int32(0))
            psi2_d = _cufft_irfft(psi2_k_d, n)
            CUDA.unsafe_free!(psi2_k_d)
            psi2_host[dim] = Array(psi2_d)
            CUDA.unsafe_free!(psi2_d)
        end
        CUDA.unsafe_free!(src2_k_d)
        return psi2_host
    end
end

function Huntian.compute_laplacian_gpu(delta_tile::AbstractArray{<:Real,3},
                                          nmesh::Integer, boxsize_local::Real;
                                          threads::Integer=256,
                                          return_device::Bool=false)
    n = Int(nmesh)
    @assert size(delta_tile) == (n, n, n)
    dk = 2π / Float64(boxsize_local)

    delta_d_is_input = delta_tile isa CuArray{Float32,3}
    delta_d = delta_d_is_input ? delta_tile : CuArray{Float32}(delta_tile)
    delta_k_d = _cufft_rfft(delta_d)
    if !delta_d_is_input
        CUDA.unsafe_free!(delta_d)
    end

    nk = size(delta_k_d, 1)
    total = nk * n * n
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _periodic_kernel!(
        delta_k_d, Float64(dk), Int32(n),
        Int32(4), Int32(0), Int32(0))

    lapd_d = _cufft_irfft(delta_k_d, n)
    CUDA.unsafe_free!(delta_k_d)
    if return_device
        return lapd_d
    else
        out = Array(lapd_d)
        CUDA.unsafe_free!(lapd_d)
        return out
    end
end

# Peak finding: GPU `Filters.smooth_field` + `PeakFind.find_peaks` on the
# periodic tile. Unlike `_periodic_kernel!`, Nyquist planes are not zeroed
# (the window is real, so Hermitian symmetry is preserved), matching CPU.
# wsmooth: 0 = Gaussian (Fortran convention, kR/2), 1 = top-hat,
#          3 = k²-weighted Gaussian (for lapd).
function _smooth_window_kernel!(arr_k, dk::Float64, Rf::Float64, n::Int32,
                                  wsmooth::Int32)
    nk = (n >> _I1) + _I1
    nyq = n >> _I1
    total = Int64(nk) * Int64(n) * Int64(n)
    idx0 = Int64(blockIdx().x - _I1) * Int64(blockDim().x) + Int64(threadIdx().x)
    if idx0 > total
        return
    end
    li = idx0 - Int64(1)
    ix = Int32(li % Int64(nk)) + _I1
    rem = li ÷ Int64(nk)
    iy = Int32(rem % Int64(n)) + _I1
    iz = Int32(rem ÷ Int64(n)) + _I1

    iy_signed = iy <= (nyq + _I1) ? (iy - _I1) : (iy - n - _I1)
    iz_signed = iz <= (nyq + _I1) ? (iz - _I1) : (iz - n - _I1)
    kx = Float64(ix - _I1) * dk
    ky = Float64(iy_signed) * dk
    kz = Float64(iz_signed) * dk
    k2 = kx * kx + ky * ky + kz * kz
    # k=0 left unchanged, as on CPU
    if k2 == 0.0
        return
    end
    k = sqrt(k2)
    kR = k * Rf
    w = 0.0
    if wsmooth == _I0
        # W = exp(-(kR/2)²/2)
        w = exp(-kR * kR * 0.125)
    elseif wsmooth == Int32(3)
        w = exp(-kR * kR * 0.125) * k2
    else
        # top-hat
        w = 3.0 * (sin(kR) - kR * cos(kR)) / (kR * kR * kR)
    end
    @inbounds val = arr_k[ix, iy, iz]
    @inbounds arr_k[ix, iy, iz] = val * Float32(w)
    return
end

# Local-maximum finder, one thread per cell of the inner cube (nbuff+1 : n-nbuff)³.
# Output slots are reserved with an atomic counter; peaks beyond max_peaks are
# not stored (the host truncates the count to max_peaks).
function _find_peaks_kernel!(peak_count, peak_i, peak_j, peak_k,
                               peak_delta, peak_lapd, peak_filter_idx,
                               mask_arr, delta_s, lapd_s,
                               n::Int32, nbuff::Int32, fcrit::Float32,
                               has_lapd::Int32, filter_idx::Int32,
                               max_peaks::Int32)
    inner_n = n - Int32(2) * nbuff
    total = Int64(inner_n) * Int64(inner_n) * Int64(inner_n)
    idx0 = Int64(blockIdx().x - _I1) * Int64(blockDim().x) + Int64(threadIdx().x)
    if idx0 > total
        return
    end
    li = idx0 - Int64(1)
    i_inner = Int32(li % Int64(inner_n)) + _I1
    rem = li ÷ Int64(inner_n)
    j_inner = Int32(rem % Int64(inner_n)) + _I1
    k_inner = Int32(rem ÷ Int64(inner_n)) + _I1
    i = i_inner + nbuff
    j = j_inner + nbuff
    k = k_inner + nbuff

    @inbounds if mask_arr[i, j, k] == Int8(1)
        return
    end
    @inbounds ff = delta_s[i, j, k]
    if ff < fcrit
        return
    end

    # No cell in the 3×3×3 stencil may exceed ff (ties allowed); including
    # the centre is harmless. Same test as CPU.
    is_max = true
    for kk in (k - _I1):(k + _I1)
        for jj in (j - _I1):(j + _I1)
            for ii in (i - _I1):(i + _I1)
                @inbounds if ff < delta_s[ii, jj, kk]
                    is_max = false
                end
            end
        end
    end
    if !is_max
        return
    end

    old = CUDA.atomic_add!(CUDA.pointer(peak_count, 1), Int32(1))
    slot = old + _I1
    if slot > max_peaks
        return
    end
    @inbounds mask_arr[i, j, k] = Int8(1)
    @inbounds peak_i[slot] = i
    @inbounds peak_j[slot] = j
    @inbounds peak_k[slot] = k
    @inbounds peak_delta[slot] = ff
    @inbounds peak_filter_idx[slot] = filter_idx
    if has_lapd == _I1
        @inbounds peak_lapd[slot] = lapd_s[i, j, k]
    end
    return
end

function Huntian.peak_find_tile_gpu(delta_tile::AbstractArray{<:Real,3},
                                        filters::AbstractVector,
                                        fcrits::AbstractVector{<:Real},
                                        tile_masks_in::AbstractArray{<:Integer,3},
                                        xbx::Real, ybx::Real, zbx::Real,
                                        alatt::Real, nbuff::Integer,
                                        wsmooth::Integer, ioutshear::Integer;
                                        threads::Integer=256,
                                        max_peaks::Integer=2_000_000)
    n = Int(size(delta_tile, 1))
    @assert size(delta_tile) == (n, n, n)
    @assert size(tile_masks_in) == (n, n, n)
    @assert length(filters) == length(fcrits)
    dk = 2π / (n * Float64(alatt))    # periodic: boxsize_local = n * alatt
    has_lapd = ioutshear >= 1 ? 1 : 0

    # A CuArray input is used in place and not freed. FFT once for all filters.
    delta_d_is_input = delta_tile isa CuArray{Float32,3}
    delta_d = delta_d_is_input ? delta_tile : CuArray{Float32}(delta_tile)
    delta_k_d = _cufft_rfft(delta_d)
    if !delta_d_is_input
        CUDA.unsafe_free!(delta_d)
    end

    # mask accumulates across filters
    mask_d = CuArray(Int8.(tile_masks_in))

    peak_count_d = CUDA.zeros(Int32, 1)
    peak_i_d     = CUDA.zeros(Int32, max_peaks)
    peak_j_d     = CUDA.zeros(Int32, max_peaks)
    peak_k_d     = CUDA.zeros(Int32, max_peaks)
    peak_delta_d = CUDA.zeros(Float32, max_peaks)
    peak_filter_d = CUDA.zeros(Int32, max_peaks)
    peak_lapd_d  = has_lapd == 1 ? CUDA.zeros(Float32, max_peaks) :
                                    CUDA.zeros(Float32, 1)   # dummy, unused

    nk_freq = size(delta_k_d, 1)
    smooth_total = nk_freq * n * n
    smooth_blocks = cld(smooth_total, threads)

    inner_n = n - 2 * Int(nbuff)
    find_total = inner_n * inner_n * inner_n
    find_blocks = cld(find_total, threads)

    smoothed_k_d = similar(delta_k_d)
    lapd_k_d     = has_lapd == 1 ? similar(delta_k_d) : delta_k_d  # alias if unused

    for (fi, (filter, fcrit)) in enumerate(zip(filters, fcrits))
        Rf = Float64(filter[3])
        copyto!(smoothed_k_d, delta_k_d)
        @cuda threads=threads blocks=smooth_blocks _smooth_window_kernel!(
            smoothed_k_d, dk, Rf, Int32(n), Int32(wsmooth))
        delta_s_d = _cufft_irfft(smoothed_k_d, n)

        lapd_s_d = if has_lapd == 1
            copyto!(lapd_k_d, delta_k_d)
            @cuda threads=threads blocks=smooth_blocks _smooth_window_kernel!(
                lapd_k_d, dk, Rf, Int32(n), Int32(3))
            _cufft_irfft(lapd_k_d, n)
        else
            delta_s_d  # dummy, kernel won't read
        end

        @cuda threads=threads blocks=find_blocks _find_peaks_kernel!(
            peak_count_d, peak_i_d, peak_j_d, peak_k_d,
            peak_delta_d, peak_lapd_d, peak_filter_d,
            mask_d, delta_s_d, lapd_s_d,
            Int32(n), Int32(nbuff), Float32(fcrit),
            Int32(has_lapd), Int32(fi), Int32(max_peaks))

        CUDA.unsafe_free!(delta_s_d)
        if has_lapd == 1
            CUDA.unsafe_free!(lapd_s_d)
        end
    end

    CUDA.unsafe_free!(smoothed_k_d)
    if has_lapd == 1
        CUDA.unsafe_free!(lapd_k_d)
    end
    CUDA.unsafe_free!(delta_k_d)

    npeaks = Int(Array(peak_count_d)[1])
    npeaks = min(npeaks, max_peaks)
    peak_i_h     = Array(peak_i_d[1:npeaks])
    peak_j_h     = Array(peak_j_d[1:npeaks])
    peak_k_h     = Array(peak_k_d[1:npeaks])
    peak_delta_h = Array(peak_delta_d[1:npeaks])
    peak_filter_h = Array(peak_filter_d[1:npeaks])
    peak_lapd_h  = has_lapd == 1 ? Array(peak_lapd_d[1:npeaks]) : Float32[]
    mask_out = Array(mask_d)

    CUDA.unsafe_free!(peak_count_d)
    CUDA.unsafe_free!(peak_i_d)
    CUDA.unsafe_free!(peak_j_d)
    CUDA.unsafe_free!(peak_k_d)
    CUDA.unsafe_free!(peak_delta_d)
    CUDA.unsafe_free!(peak_filter_d)
    CUDA.unsafe_free!(peak_lapd_d)
    CUDA.unsafe_free!(mask_d)

    # Cell centres as in the CPU peak finder
    cen1 = 0.5 * (n + 1)
    peaks = Vector{Huntian.PeakCandidate}(undef, npeaks)
    Rf_out  = Vector{Float64}(undef, npeaks)
    FcRf_out = Vector{Float32}(undef, npeaks)
    d2Rf_out = Vector{Float32}(undef, npeaks)
    for p in 1:npeaks
        i = Int(peak_i_h[p]); j = Int(peak_j_h[p]); k = Int(peak_k_h[p])
        xc = Float64(xbx) + Float64(alatt) * (i - cen1)
        yc = Float64(ybx) + Float64(alatt) * (j - cen1)
        zc = Float64(zbx) + Float64(alatt) * (k - cen1)
        ipp = i + (j - 1) * n + (k - 1) * n * n
        fi_ = Int(peak_filter_h[p])
        Rf_out[p] = Float64(filters[fi_][3])
        FcRf_out[p] = peak_delta_h[p]
        d2Rf_out[p] = has_lapd == 1 ? peak_lapd_h[p] : 0.0f0
        peaks[p] = Huntian.PeakCandidate{Float32}(
            i, j, k, ipp, xc, yc, zc, peak_delta_h[p], Rf_out[p])
    end

    return (peaks=peaks, Rf=Rf_out, FcRf=FcRf_out, d2Rf=d2Rf_out, mask=mask_out)
end

# Coarse → fine tile tricubic (Catmull-Rom) interpolation; GPU version of
# `MultiResolution._interpolate_to_tile`, periodic in both grids.

@inline function _catmull_rom_f32(t::Float32)
    t2 = t * t; t3 = t2 * t
    w0 = -0.5f0 * t3 + t2 - 0.5f0 * t
    w1 =  1.5f0 * t3 - 2.5f0 * t2 + 1.0f0
    w2 = -1.5f0 * t3 + 2.0f0 * t2 + 0.5f0 * t
    w3 =  0.5f0 * t3 - 0.5f0 * t2
    return (w0, w1, w2, w3)
end

@inline function _mod1_i32(i::Int32, M::Int32)
    # mod1(i, M)
    r = (i - _I1) % M
    if r < _I0
        r += M
    end
    return r + _I1
end

function _tricubic_kernel!(tile_out, coarse, i0::Int32, j0::Int32, k0::Int32,
                             block::Int32, nmesh::Int32, N::Int32, M::Int32)
    total = Int64(nmesh) * Int64(nmesh) * Int64(nmesh)
    idx0 = Int64(blockIdx().x - _I1) * Int64(blockDim().x) + Int64(threadIdx().x)
    if idx0 > total
        return
    end
    li_ = idx0 - Int64(1)
    li = Int32(li_ % Int64(nmesh)) + _I1
    rem = li_ ÷ Int64(nmesh)
    lj = Int32(rem % Int64(nmesh)) + _I1
    lk = Int32(rem ÷ Int64(nmesh)) + _I1

    # Wrapped global fine index → fractional coarse coordinate (cell-centred).
    gi = _mod1_i32(i0 + li - _I1, N)
    gj = _mod1_i32(j0 + lj - _I1, N)
    gk = _mod1_i32(k0 + lk - _I1, N)

    # Float64 as on CPU
    fx = (Float64(gi) - 0.5) / Float64(block) + 0.5
    fy = (Float64(gj) - 0.5) / Float64(block) + 0.5
    fz = (Float64(gk) - 0.5) / Float64(block) + 0.5

    ix = Int32(floor(fx)); dx_ = Float32(fx - Float64(ix))
    iy = Int32(floor(fy)); dy_ = Float32(fy - Float64(iy))
    iz = Int32(floor(fz)); dz_ = Float32(fz - Float64(iz))

    (wx0, wx1, wx2, wx3) = _catmull_rom_f32(dx_)
    (wy0, wy1, wy2, wy3) = _catmull_rom_f32(dy_)
    (wz0, wz1, wz2, wz3) = _catmull_rom_f32(dz_)

    ix_arr = (_mod1_i32(ix - _I1, M), _mod1_i32(ix, M),
              _mod1_i32(ix + _I1, M), _mod1_i32(ix + Int32(2), M))
    iy_arr = (_mod1_i32(iy - _I1, M), _mod1_i32(iy, M),
              _mod1_i32(iy + _I1, M), _mod1_i32(iy + Int32(2), M))
    iz_arr = (_mod1_i32(iz - _I1, M), _mod1_i32(iz, M),
              _mod1_i32(iz + _I1, M), _mod1_i32(iz + Int32(2), M))
    wx = (wx0, wx1, wx2, wx3)
    wy = (wy0, wy1, wy2, wy3)
    wz = (wz0, wz1, wz2, wz3)

    val = 0.0f0
    @inbounds for kk_idx in 1:4
        kk = iz_arr[kk_idx]
        wk = wz[kk_idx]
        for jj_idx in 1:4
            jj = iy_arr[jj_idx]
            wj = wy[jj_idx]
            for ii_idx in 1:4
                ii = ix_arr[ii_idx]
                wi = wx[ii_idx]
                val += wi * wj * wk * coarse[ii, jj, kk]
            end
        end
    end
    @inbounds tile_out[li, lj, lk] = val
    return
end

function Huntian.interpolate_to_tile_gpu(coarse::AbstractArray{<:Real,3},
                                             it::Integer, jt::Integer, kt::Integer,
                                             nsub::Integer, nmesh::Integer,
                                             N::Integer, M::Integer;
                                             threads::Integer=256,
                                             return_device::Bool=false)
    @assert size(coarse) == (M, M, M)
    block = Int(N) ÷ Int(M)
    nbuff = (Int(nmesh) - Int(nsub)) ÷ 2
    i0 = (Int(it) - 1) * Int(nsub) + 1 - nbuff
    j0 = (Int(jt) - 1) * Int(nsub) + 1 - nbuff
    k0 = (Int(kt) - 1) * Int(nsub) + 1 - nbuff

    coarse_d_is_input = coarse isa CuArray{Float32,3}
    coarse_d = coarse_d_is_input ? coarse : CuArray{Float32}(coarse)
    tile_d = CUDA.zeros(Float32, Int(nmesh), Int(nmesh), Int(nmesh))

    total = Int(nmesh)^3
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _tricubic_kernel!(
        tile_d, coarse_d, Int32(i0), Int32(j0), Int32(k0),
        Int32(block), Int32(nmesh), Int32(N), Int32(M))

    if !coarse_d_is_input
        CUDA.unsafe_free!(coarse_d)
    end
    if return_device
        return tile_d
    else
        out = Array(tile_d)
        CUDA.unsafe_free!(tile_d)
        return out
    end
end

# Bind the current task to CUDA device `id` (0-based).
function Huntian.set_cuda_device!(id::Int)
    CUDA.device!(id)
    return nothing
end

# Field-matter lightcone painting: on-device 2LPT displacement, HEALPix RING
# pixelization and atomic accumulation into device-resident Float64 maps
# (npix × nk; Float64 atomics need sm_60+). One thread per core cell, which
# subdivides itself into ns³ sub-cells so a cell spans at most ~one pixel.

import Huntian.MultiResolution: ang2pix_ring, pix2vec_ring, _isw_ray, _isw_tile_cone

function _fieldmap_paint_kernel!(maps, p1x, p1y, p1z, p2x, p2y, p2z, rt,
                                 nmesh::Int, nbuff::Int, alatt::Float64,
                                 xbx::Float64, ybx::Float64, zbx::Float64,
                                 ox::Float64, oy::Float64, oz::Float64,
                                 rmin::Float64, chimax::Float64, inv_dr::Float64,
                                 nrt::Int, theta_pix::Float64, subdiv_max::Int,
                                 nside::Int, nk::Int, has2::Bool,
                                 vwmask::UInt32, excl, has_excl::Bool,
                                 pot, has_pot::Bool, pwmask::UInt32)
    ncore = nmesh - 2 * nbuff
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > ncore * ncore * ncore && return nothing
    t = idx - 1
    i = nbuff + 1 + t % ncore
    j = nbuff + 1 + (t ÷ ncore) % ncore
    k = nbuff + 1 + t ÷ (ncore * ncore)
    if has_excl
        @inbounds excl[i-nbuff, j-nbuff, k-nbuff] && return nothing
    end
    cen = 0.5 * (nmesh + 1)
    qx = xbx + alatt * (i - cen)
    qy = ybx + alatt * (j - cen)
    qz = zbx + alatt * (k - cen)
    dqx = qx - ox; dqy = qy - oy; dqz = qz - oz
    rq = sqrt(dqx * dqx + dqy * dqy + dqz * dqz)
    (rmin <= rq <= chimax) || return nothing
    @inbounds begin
        s1x = Float64(p1x[i, j, k]); s1y = Float64(p1y[i, j, k]); s1z = Float64(p1z[i, j, k])
        s2x = 0.0; s2y = 0.0; s2z = 0.0
        if has2
            s2x = Float64(p2x[i, j, k]); s2y = Float64(p2y[i, j, k]); s2z = Float64(p2z[i, j, k])
        end
        pv = has_pot ? Float64(pot[i, j, k]) : 0.0
        ns = min(subdiv_max, max(1, ceil(Int, (alatt / rq) / theta_pix)))
        wsub = 1.0 / (ns * ns * ns)
        for c3 in 1:ns, c2i in 1:ns, c1 in 1:ns
            qsx = qx + ((c1 - 0.5) / ns - 0.5) * alatt
            qsy = qy + ((c2i - 0.5) / ns - 0.5) * alatt
            qsz = qz + ((c3 - 0.5) / ns - 0.5) * alatt
            dsx = qsx - ox; dsy = qsy - oy; dsz = qsz - oz
            rqs = sqrt(dsx * dsx + dsy * dsy + dsz * dsz)
            (rmin <= rqs <= chimax) || continue
            xr = rqs * inv_dr + 1.0
            ii = unsafe_trunc(Int, xr)
            ii = ii < 1 ? 1 : (ii > nrt - 1 ? nrt - 1 : ii)
            tt = xr - ii
            D  = rt[ii, 1] * (1.0 - tt) + rt[ii+1, 1] * tt
            c2 = rt[ii, 2] * (1.0 - tt) + rt[ii+1, 2] * tt
            ex = qsx + D * s1x + c2 * s2x - ox
            ey = qsy + D * s1y + c2 * s2y - oy
            ez = qsz + D * s1z + c2 * s2z - oz
            pix = ang2pix_ring(nside, ex, ey, ez)
            pixq = pwmask != UInt32(0) ? ang2pix_ring(nside, dsx, dsy, dsz) : pix
            vr = 0.0
            if vwmask != UInt32(0)
                vf = rt[ii, 3] * (1.0 - tt) + rt[ii+1, 3] * tt
                vr = vf * ((D * s1x + 2 * c2 * s2x) * dsx + (D * s1y + 2 * c2 * s2y) * dsy +
                           (D * s1z + 2 * c2 * s2z) * dsz) / rqs
            end
            for ik in 1:nk
                w = (rt[ii, 3+ik] * (1.0 - tt) + rt[ii+1, 3+ik] * tt) * wsub
                (vwmask >> (ik - 1)) & UInt32(1) == UInt32(1) && (w *= vr)
                if (pwmask >> (ik - 1)) & UInt32(1) == UInt32(1)
                    # potential-weighted kernels deposit at the Lagrangian pixel (see src/FieldMap.jl)
                    CUDA.@atomic maps[pixq, ik] += w * pv
                else
                    CUDA.@atomic maps[pix, ik] += w
                end
            end
        end
    end
    return nothing
end

# ISW line-of-sight gather (isw_mode=:gather): one thread per pixel of the tile's cone; each pixel
# gets exactly one ray segment per tile, and tiles on one device run in stream order, so a plain
# add is race-free (atomic kept for safety; contention-free).
function _isw_gather_kernel!(maps, ik::Int, pot, wl, nside::Int, plo::Int, phi::Int,
                             cx::Float64, cy::Float64, cz::Float64, cosmax::Float64,
                             nmesh::Int, nbuff::Int, alatt::Float64,
                             xbx::Float64, ybx::Float64, zbx::Float64,
                             ox::Float64, oy::Float64, oz::Float64,
                             rmin::Float64, chimax::Float64, inv_dr::Float64, nrt::Int, h::Float64)
    p = plo + (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    p > phi && return nothing
    nx, ny, nz = pix2vec_ring(nside, p)
    nx * cx + ny * cy + nz * cz >= cosmax || return nothing
    s = _isw_ray(pot, wl, nx, ny, nz, nmesh, nbuff, alatt, xbx, ybx, zbx, ox, oy, oz,
                 rmin, chimax, inv_dr, nrt, h)
    s != 0.0 && CUDA.@atomic maps[p, ik] += s
    return nothing
end

function Huntian.gather_tile_isw_gpu!(maps_d::CuArray{Float64,2}, ik::Int, pot::CuArray{Float32,3},
        wl::Vector{Float64}, nside::Int, nmesh::Int, nbuff::Int, alatt::Float64,
        xbx::Float64, ybx::Float64, zbx::Float64, obs::NTuple{3,Float64},
        rmin::Float64, chimax::Float64, inv_dr::Float64, nrt::Int, h::Float64)
    plo, phi, cx, cy, cz, cosmax = _isw_tile_cone(nside, nmesh, nbuff, alatt, xbx, ybx, zbx, obs)
    wl_d = CuArray(wl)
    threads = 256
    @cuda threads=threads blocks=cld(phi - plo + 1, threads) _isw_gather_kernel!(
        maps_d, ik, pot, wl_d, nside, plo, phi, cx, cy, cz, cosmax, nmesh, nbuff, alatt,
        xbx, ybx, zbx, obs[1], obs[2], obs[3], rmin, chimax, inv_dr, nrt, h)
    CUDA.synchronize()
    CUDA.unsafe_free!(wl_d)
    return nothing
end

# Allocation/collection helpers so src/FieldMap.jl never touches CUDA types.
Huntian.fieldmap_gpu_alloc(npix::Int, nk::Int, rtM::Matrix{Float64}) =
    (CUDA.zeros(Float64, npix, nk), CuArray(rtM))
Huntian.fieldmap_gpu_collect(maps_d::CuArray{Float64,2}) = Array(maps_d)

"""
    paint_tile_field_gpu!(maps_d, p1x, p1y, p1z, p2x, p2y, p2z, rt_d, ...)

Device-side painting of one tile's core cells into `maps_d` (npix × nk Float64
CuArray, RING ordering). `rt_d` is the radial factor table (nrt × (3+nk):
columns D, coef2, vfac, then one weight column per kernel). `vwmask` bit ik-1
set means kernel ik's weight is multiplied by the cell's LOS velocity v_r [km/s].
`excl` is an optional host-side core-shaped Bool mask (true = skip cell). When
`has2=false` the `p2*` arguments are ignored (pass the `p1*` arrays as
placeholders). Kernels with bit ik-1 of `pwmask` set deposit `w * pot` at the
undisplaced (Lagrangian) pixel instead.
"""
function Huntian.paint_tile_field_gpu!(maps_d::CuArray{Float64,2},
        p1x::CuArray{Float32,3}, p1y::CuArray{Float32,3}, p1z::CuArray{Float32,3},
        p2x::CuArray{Float32,3}, p2y::CuArray{Float32,3}, p2z::CuArray{Float32,3},
        rt_d::CuArray{Float64,2}, nmesh::Int, nbuff::Int, alatt::Float64,
        xbx::Float64, ybx::Float64, zbx::Float64, obs::NTuple{3,Float64},
        rmin::Float64, chimax::Float64, inv_dr::Float64, nrt::Int,
        theta_pix::Float64, subdiv_max::Int, nside::Int, nk::Int, has2::Bool,
        vwmask::UInt32=UInt32(0), excl::Union{Nothing,Array{Bool,3}}=nothing,
        pot::Union{Nothing,CuArray{Float32,3}}=nothing, pwmask::UInt32=UInt32(0))
    ncore = nmesh - 2 * nbuff
    ntot = ncore^3
    has_excl = excl !== nothing
    excl_d = has_excl ? CuArray(excl) : CUDA.zeros(Bool, 1, 1, 1)
    has_pot = pot !== nothing
    pot_d = has_pot ? pot : p1x          # placeholder, never read
    threads = 256
    @cuda threads=threads blocks=cld(ntot, threads) _fieldmap_paint_kernel!(
        maps_d, p1x, p1y, p1z, p2x, p2y, p2z, rt_d, nmesh, nbuff, alatt,
        xbx, ybx, zbx, obs[1], obs[2], obs[3], rmin, chimax, inv_dr, nrt,
        theta_pix, subdiv_max, nside, nk, has2, vwmask, excl_d, has_excl,
        pot_d, has_pot, pwmask)
    CUDA.synchronize()
    return nothing
end

end # module CUDAExt
