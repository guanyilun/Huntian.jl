# MPI + PencilFFTs extension: distributed field generation and multi-tile halo finding
# (`run_multitile_mpi`).
module MPIExt

using Huntian
using MPI
using PencilFFTs
using PencilFFTs.Transforms: RFFT, RFFT!
using PencilArrays
using PencilArrays: Transpositions
using FFTW

# MPI.Buffer.count is Cint, limiting messages to 2^31-1 elements; large pencil transposes
# (e.g. 6144³ on few ranks) exceed this. Use a contiguous derived datatype so that
# count = nelements ÷ chunk fits in Cint.

"""Find smallest divisor of `n` such that `n ÷ divisor ≤ typemax(Cint)`."""
function _large_count_chunk(n::Integer)
    max_count = Int(typemax(Cint))
    n ≤ max_count && return 1
    m = cld(n, max_count)
    while n % m != 0
        m += 1
    end
    return m
end

# Replacement for PencilArrays' internal `Transpositions.mpi_buffer`, whose
# MPI.Buffer(view) does Cint(length(view)) and fails above 2^31 elements.
# Installed in __init__: overwriting a method during precompilation is an error.
function _mpi_buffer_large(buf::AbstractArray, off, len)
    inds = (off + 1):(off + len)
    v = view(buf, inds)
    if len ≤ typemax(Cint)
        return MPI.Buffer(v)
    end
    chunk = _large_count_chunk(len)
    bigtype = MPI.Types.create_contiguous(Cint(chunk), MPI.Datatype(eltype(buf)))
    MPI.Types.commit!(bigtype)
    return MPI.Buffer(v, Cint(len ÷ chunk), bigtype)
end

"""Factor nranks into (p1, p2) for 2D pencil decomposition."""
function _factor_procs(nranks::Int)
    p1 = isqrt(nranks)
    while nranks % p1 != 0
        p1 -= 1
    end
    return (p1, nranks ÷ p1)
end

"""Block-assign ntile³ tiles to `nranks` MPI ranks.
Returns the list of (it,jt,kt) tuples assigned to `rank` (0-based)."""
function _assign_tiles(ntile::Int, nranks::Int, rank::Int)
    total = ntile^3
    tiles_per_rank = cld(total, nranks)
    start_idx = rank * tiles_per_rank + 1
    end_idx = min((rank + 1) * tiles_per_rank, total)

    tile_ids = NTuple{3,Int}[]
    idx = 0
    for kt in 1:ntile, jt in 1:ntile, it in 1:ntile
        idx += 1
        if start_idx <= idx <= end_idx
            push!(tile_ids, (it, jt, kt))
        end
    end
    return tile_ids
end

"""Fill a PencilArray from a full N³ array (all ranks must have `full_arr`)."""
function _fill_pencil!(pencil_arr, full_arr)
    gv = PencilArrays.global_view(pencil_arr)
    pen = PencilArrays.pencil(pencil_arr)
    ranges = PencilArrays.range_local(pen)
    for k in ranges[3], j in ranges[2], i in ranges[1]
        gv[i, j, k] = full_arr[i, j, k]
    end
    return pencil_arr
end

"""Extract tile subcubes from a pencil-distributed real-space field.

Each rank receives only the nmesh³ subcubes for its own tiles via point-to-point
MPI (Isend/Irecv), instead of gathering the full N³ field.  Memory per rank is
O(ntile_local × nmesh³) instead of O(N³).

When `global_tile_list` is provided, only those tiles participate in communication
(for batched extraction). All ranks must pass the same `global_tile_list`.

Returns `Dict{NTuple{3,Int}, Array{T,3}}` mapping tile IDs to nmesh³ arrays.
"""
function _extract_my_tiles(pencil_arr, my_tiles::Vector{NTuple{3,Int}},
                           ntile::Int, nsub::Int, nmesh::Int,
                           ::Type{T}, comm::MPI.Comm;
                           global_tile_list::Union{Nothing, Vector{NTuple{3,Int}}}=nothing) where T
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    pen = PencilArrays.pencil(pencil_arr)
    local_ranges = PencilArrays.range_local(pen)
    gv = PencilArrays.global_view(pencil_arr)

    # Share pencil ranges across all ranks
    my_flat = Int32[first(local_ranges[1]), last(local_ranges[1]),
                    first(local_ranges[2]), last(local_ranges[2]),
                    first(local_ranges[3]), last(local_ranges[3])]
    all_flat = MPI.Allgather(my_flat, comm)
    all_ranges_mat = reshape(all_flat, 6, nranks)

    # Tile owner map (deterministic, all ranks agree)
    tile_owner = Dict{NTuple{3,Int}, Int}()
    for r in 0:nranks-1
        for tid in _assign_tiles(ntile, nranks, r)
            tile_owner[tid] = r
        end
    end

    # Canonical tile list: identical ordering on all ranks gives consistent MPI tags
    if global_tile_list === nothing
        all_tiles = NTuple{3,Int}[]
        for kt in 1:ntile, jt in 1:ntile, it in 1:ntile
            push!(all_tiles, (it, jt, kt))
        end
    else
        all_tiles = global_tile_list
    end

    result = Dict{NTuple{3,Int}, Array{T,3}}()
    for tid in my_tiles
        result[tid] = Array{T,3}(undef, nmesh, nmesh, nmesh)
    end

    send_reqs = MPI.Request[]
    send_bufs = Vector{T}[]
    recv_reqs = MPI.Request[]
    recv_bufs = Vector{T}[]
    recv_meta = Tuple{NTuple{3,Int}, UnitRange{Int}, UnitRange{Int}, UnitRange{Int}}[]

    for (tidx, tid) in enumerate(all_tiles)
        it, jt, kt = tid
        gi1 = (it-1)*nsub + 1; gi2 = gi1 + nmesh - 1
        gj1 = (jt-1)*nsub + 1; gj2 = gj1 + nmesh - 1
        gk1 = (kt-1)*nsub + 1; gk2 = gk1 + nmesh - 1
        owner = tile_owner[tid]

        i_ovlp = max(gi1, first(local_ranges[1])):min(gi2, last(local_ranges[1]))
        j_ovlp = max(gj1, first(local_ranges[2])):min(gj2, last(local_ranges[2]))
        k_ovlp = max(gk1, first(local_ranges[3])):min(gk2, last(local_ranges[3]))
        has_data = !isempty(i_ovlp) && !isempty(j_ovlp) && !isempty(k_ovlp)

        if has_data && owner == rank
            # Local copy
            tile = result[tid]
            for gk in k_ovlp, gj in j_ovlp, gi in i_ovlp
                tile[gi - gi1 + 1, gj - gj1 + 1, gk - gk1 + 1] = T(gv[gi, gj, gk])
            end
        elseif has_data && owner != rank
            buf = Vector{T}(undef, length(i_ovlp) * length(j_ovlp) * length(k_ovlp))
            idx = 0
            for gk in k_ovlp, gj in j_ovlp, gi in i_ovlp
                idx += 1
                buf[idx] = T(gv[gi, gj, gk])
            end
            push!(send_bufs, buf)
            push!(send_reqs, MPI.Isend(buf, owner, tidx, comm))
        end

        if owner == rank
            for src in 0:nranks-1
                src == rank && continue
                off = src * 6
                sr = (Int(all_ranges_mat[1, src+1]):Int(all_ranges_mat[2, src+1]),
                      Int(all_ranges_mat[3, src+1]):Int(all_ranges_mat[4, src+1]),
                      Int(all_ranges_mat[5, src+1]):Int(all_ranges_mat[6, src+1]))
                si = max(gi1, first(sr[1])):min(gi2, last(sr[1]))
                sj = max(gj1, first(sr[2])):min(gj2, last(sr[2]))
                sk = max(gk1, first(sr[3])):min(gk2, last(sr[3]))

                if !isempty(si) && !isempty(sj) && !isempty(sk)
                    buf = Vector{T}(undef, length(si) * length(sj) * length(sk))
                    push!(recv_bufs, buf)
                    push!(recv_reqs, MPI.Irecv!(buf, src, tidx, comm))
                    push!(recv_meta, (tid,
                                      si .- gi1 .+ 1,
                                      sj .- gj1 .+ 1,
                                      sk .- gk1 .+ 1))
                end
            end
        end
    end

    MPI.Waitall(recv_reqs)
    for (idx, (tid, di_rng, dj_rng, dk_rng)) in enumerate(recv_meta)
        tile = result[tid]
        buf = recv_bufs[idx]
        buf_idx = 0
        for dk in dk_rng, dj in dj_rng, di in di_rng
            buf_idx += 1
            tile[di, dj, dk] = buf[buf_idx]
        end
    end

    # send_bufs must stay alive until the sends complete
    MPI.Waitall(send_reqs)

    return result
end

"""Convert 1-based flat index to (i,j,k) for column-major n×n×n array."""
function _ipp_to_ijk(ipp::Int, n::Int)
    k = (ipp - 1) ÷ (n * n) + 1
    rem = (ipp - 1) % (n * n)
    j = rem ÷ n + 1
    i = rem % n + 1
    return (i, j, k)
end

# Distributed k-space operations. k-grids use rfftfreq along x (first, halved
# dimension) and fftfreq along y, z.

"""Multiply PencilArray delta_k by √(P(k) dk³ N³) in-place (distributed)."""
function _distributed_convolve_pk!(delta_k_pencil, pk, N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = FFTW.rfftfreq(N, N * dk)
    ky_arr = FFTW.fftfreq(N, N * dk)
    kz_arr = FFTW.fftfreq(N, N * dk)

    pen = PencilArrays.pencil(delta_k_pencil)
    ranges = PencilArrays.range_local(pen)
    gv = PencilArrays.global_view(delta_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        if k2 == 0.0
            gv[gix, giy, giz] = 0.0
            continue
        end
        k = sqrt(k2)
        amp = sqrt(pk(k) * dk^3 * N^3)
        gv[gix, giy, giz] *= amp
    end
end

"""Apply a k-space window function to a copy of delta_k (distributed).
Returns a new PencilArray with window applied."""
function _distributed_apply_window(delta_k_pencil, N::Int, boxsize::Float64,
                                   Rf::Float64, wsmooth::Int)
    dk = 2π / boxsize
    kx_arr = FFTW.rfftfreq(N, N * dk)
    ky_arr = FFTW.fftfreq(N, N * dk)
    kz_arr = FFTW.fftfreq(N, N * dk)

    smoothed_k = similar(delta_k_pencil)
    smoothed_k .= delta_k_pencil

    pen = PencilArrays.pencil(smoothed_k)
    ranges = PencilArrays.range_local(pen)
    gv = PencilArrays.global_view(smoothed_k)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k = sqrt(kx^2 + ky^2 + kz^2)
        fkR = k * Rf
        fkR == 0.0 && continue
        if wsmooth == 0
            w = Huntian.gaussian_window_fortran(fkR)
        elseif wsmooth == 3
            w = Huntian.gaussian_window_fortran(fkR) * k^2
        else
            w = Huntian.tophat_window(fkR)
        end
        gv[gix, giy, giz] *= w
    end
    return smoothed_k
end

"""Compute 1LPT displacement component `dim` (1=x,2=y,3=z) from delta_k.
Returns a k-space PencilArray ready for inverse FFT."""
function _distributed_1lpt_component(delta_k_pencil, dim::Int,
                                     N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = Float64.(FFTW.rfftfreq(N, N * dk))
    ky_arr = Float64.(FFTW.fftfreq(N, N * dk))
    kz_arr = Float64.(FFTW.fftfreq(N, N * dk))
    nyq = N ÷ 2
    nk = N ÷ 2 + 1

    psi_k = similar(delta_k_pencil)
    pen = PencilArrays.pencil(psi_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(psi_k)
    gv_in = PencilArrays.global_view(delta_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        if k2 == 0.0 || gix == nk || giy == nyq + 1 || giz == nyq + 1
            gv_out[gix, giy, giz] = 0.0
            continue
        end
        ki = dim == 1 ? kx : dim == 2 ? ky : kz
        gv_out[gix, giy, giz] = im * ki / k2 * gv_in[gix, giy, giz]
    end
    return psi_k
end

"""Compute 2LPT second-derivative potential component phi_ij(k) = -ki*kj/k² δ(k).
`di`, `dj` are dimension indices (1=x, 2=y, 3=z)."""
function _distributed_phi_ij_component(delta_k_pencil, di::Int, dj::Int,
                                       N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = Float64.(FFTW.rfftfreq(N, N * dk))
    ky_arr = Float64.(FFTW.fftfreq(N, N * dk))
    kz_arr = Float64.(FFTW.fftfreq(N, N * dk))
    nyq = N ÷ 2
    nk = N ÷ 2 + 1

    phi_k = similar(delta_k_pencil)
    pen = PencilArrays.pencil(phi_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(phi_k)
    gv_in = PencilArrays.global_view(delta_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        if k2 == 0.0 || gix == nk || giy == nyq + 1 || giz == nyq + 1
            gv_out[gix, giy, giz] = 0.0
            continue
        end
        ki = di == 1 ? kx : di == 2 ? ky : kz
        kj = dj == 1 ? kx : dj == 2 ? ky : kz
        gv_out[gix, giy, giz] = -ki * kj / k2 * gv_in[gix, giy, giz]
    end
    return phi_k
end

"""Compute 2LPT displacement component from src2_k.
psi2_i(k) = -im * ki / k² * src2(k)."""
function _distributed_2lpt_component(src2_k_pencil, dim::Int,
                                     N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = Float64.(FFTW.rfftfreq(N, N * dk))
    ky_arr = Float64.(FFTW.fftfreq(N, N * dk))
    kz_arr = Float64.(FFTW.fftfreq(N, N * dk))
    nyq = N ÷ 2
    nk = N ÷ 2 + 1

    psi2_k = similar(src2_k_pencil)
    pen = PencilArrays.pencil(psi2_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(psi2_k)
    gv_in = PencilArrays.global_view(src2_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        if k2 == 0.0 || gix == nk || giy == nyq + 1 || giz == nyq + 1
            gv_out[gix, giy, giz] = 0.0
            continue
        end
        ki = dim == 1 ? kx : dim == 2 ? ky : kz
        gv_out[gix, giy, giz] = -im * ki / k2 * gv_in[gix, giy, giz]
    end
    return psi2_k
end

"""Compute k²-weighted field in k-space (for Laplacian diagnostic)."""
function _distributed_laplacian_k(delta_k_pencil, N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = FFTW.rfftfreq(N, N * dk)
    ky_arr = FFTW.fftfreq(N, N * dk)
    kz_arr = FFTW.fftfreq(N, N * dk)

    lapd_k = similar(delta_k_pencil)
    pen = PencilArrays.pencil(lapd_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(lapd_k)
    gv_in = PencilArrays.global_view(delta_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        gv_out[gix, giy, giz] = k2 * gv_in[gix, giy, giz]
    end
    return lapd_k
end

# In-place k-space kernels writing into a pre-allocated output, avoiding a full-size
# temporary PencilArray per call (~155 GB at 6144³).

"""Apply 1LPT kernel im*ki/k² to delta_k, writing result into output_k."""
function _apply_1lpt_kernel!(output_k, delta_k_pencil, dim::Int, N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = Float64.(FFTW.rfftfreq(N, N * dk))
    ky_arr = Float64.(FFTW.fftfreq(N, N * dk))
    kz_arr = Float64.(FFTW.fftfreq(N, N * dk))
    nyq = N ÷ 2; nk = N ÷ 2 + 1

    pen = PencilArrays.pencil(output_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(output_k)
    gv_in = PencilArrays.global_view(delta_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        if k2 == 0.0 || gix == nk || giy == nyq + 1 || giz == nyq + 1
            gv_out[gix, giy, giz] = 0
            continue
        end
        ki = dim == 1 ? kx : dim == 2 ? ky : kz
        gv_out[gix, giy, giz] = im * ki / k2 * gv_in[gix, giy, giz]
    end
end

"""Apply 2LPT kernel -im*ki/k² to src2_k, writing result into output_k."""
function _apply_2lpt_kernel!(output_k, src2_k_pencil, dim::Int, N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = Float64.(FFTW.rfftfreq(N, N * dk))
    ky_arr = Float64.(FFTW.fftfreq(N, N * dk))
    kz_arr = Float64.(FFTW.fftfreq(N, N * dk))
    nyq = N ÷ 2; nk = N ÷ 2 + 1

    pen = PencilArrays.pencil(output_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(output_k)
    gv_in = PencilArrays.global_view(src2_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        if k2 == 0.0 || gix == nk || giy == nyq + 1 || giz == nyq + 1
            gv_out[gix, giy, giz] = 0
            continue
        end
        ki = dim == 1 ? kx : dim == 2 ? ky : kz
        gv_out[gix, giy, giz] = -im * ki / k2 * gv_in[gix, giy, giz]
    end
end

"""Copy delta_k into output_k with the k=0 mode zeroed: the exact k-space form of
-Σ_i phi_ii given `_apply_phi_ij_kernel!` (which, like serial LPT.jl, zeroes only
k=0). The 2LPT trace identity needs (Σ phi_ii)² = δ² exactly: zeroing the Nyquist
planes in phi_ij but not in δ breaks it (large ψ₂ errors vs LPT.jl)."""
function _apply_trace_kernel!(output_k, delta_k_pencil, N::Int)
    pen = PencilArrays.pencil(output_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(output_k)
    gv_in = PencilArrays.global_view(delta_k_pencil)
    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        if gix == 1 && giy == 1 && giz == 1
            gv_out[gix, giy, giz] = 0
        else
            gv_out[gix, giy, giz] = gv_in[gix, giy, giz]
        end
    end
end

"""Apply phi_ij kernel -ki*kj/k² to delta_k, writing result into output_k."""
function _apply_phi_ij_kernel!(output_k, delta_k_pencil, di::Int, dj::Int,
                               N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = Float64.(FFTW.rfftfreq(N, N * dk))
    ky_arr = Float64.(FFTW.fftfreq(N, N * dk))
    kz_arr = Float64.(FFTW.fftfreq(N, N * dk))
    nyq = N ÷ 2; nk = N ÷ 2 + 1

    pen = PencilArrays.pencil(output_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(output_k)
    gv_in = PencilArrays.global_view(delta_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        if k2 == 0.0   # k=0 only, as in LPT.jl: Nyquist kept so Σ phi_ii = -δ exactly
            gv_out[gix, giy, giz] = 0
            continue
        end
        ki = di == 1 ? kx : di == 2 ? ky : kz
        kj = dj == 1 ? kx : dj == 2 ? ky : kz
        gv_out[gix, giy, giz] = -ki * kj / k2 * gv_in[gix, giy, giz]
    end
end

"""Apply smoothing window to delta_k, writing result into output_k."""
function _apply_window_kernel!(output_k, delta_k_pencil, N::Int, boxsize::Float64,
                               Rf::Float64, wsmooth::Int)
    dk = 2π / boxsize
    kx_arr = FFTW.rfftfreq(N, N * dk)
    ky_arr = FFTW.fftfreq(N, N * dk)
    kz_arr = FFTW.fftfreq(N, N * dk)

    pen = PencilArrays.pencil(output_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(output_k)
    gv_in = PencilArrays.global_view(delta_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k = sqrt(kx^2 + ky^2 + kz^2)
        fkR = k * Rf
        if fkR == 0.0
            gv_out[gix, giy, giz] = gv_in[gix, giy, giz]
            continue
        end
        if wsmooth == 0
            w = Huntian.gaussian_window_fortran(fkR)
        elseif wsmooth == 3
            w = Huntian.gaussian_window_fortran(fkR) * k^2
        else
            w = Huntian.tophat_window(fkR)
        end
        gv_out[gix, giy, giz] = w * gv_in[gix, giy, giz]
    end
end

"""Apply Laplacian kernel k² to delta_k, writing result into output_k."""
function _apply_laplacian_kernel!(output_k, delta_k_pencil, N::Int, boxsize::Float64)
    dk = 2π / boxsize
    kx_arr = FFTW.rfftfreq(N, N * dk)
    ky_arr = FFTW.fftfreq(N, N * dk)
    kz_arr = FFTW.fftfreq(N, N * dk)

    pen = PencilArrays.pencil(output_k)
    ranges = PencilArrays.range_local(pen)
    gv_out = PencilArrays.global_view(output_k)
    gv_in = PencilArrays.global_view(delta_k_pencil)

    for giz in ranges[3], giy in ranges[2], gix in ranges[1]
        kx = kx_arr[gix]; ky = ky_arr[giy]; kz = kz_arr[giz]
        k2 = kx^2 + ky^2 + kz^2
        gv_out[gix, giy, giz] = k2 * gv_in[gix, giy, giz]
    end
end

"""Run shell analysis for the peaks of one tile and append halos to the output vectors
(shared by the standard and lowmem paths). Velocities are displacements in Mpc/h;
`finalize_eulerian` converts them later."""
function _analyse_tile_shells!(halos_basic, halos_ext,
        tid, peaks, peak_Rfs, peak_FcRfs, peak_d2Rfs, mask_tile,
        delta_tile, psi_x, psi_y, psi_z,
        psi2_x, psi2_y, psi2_z, lapd_tile,
        nmesh, nbuff, alatt, ntile, dcore_box,
        obs, ievol, z_max, chi2z, growth_tables, ZZon,
        Omnr, cosmo, ct, shells, cfg, ioutshear, ilpt)

    isempty(peaks) && return

    it, jt, kt = tid
    fill!(mask_tile, 0)

    pg = Huntian.PeakGrid(delta_tile, psi_x, psi_y, psi_z,
                              psi2_x, psi2_y, psi2_z,
                              mask_tile, (nmesh, nmesh, nmesh), lapd_tile)

    for idx in 1:length(peaks)
        peak = peaks[idx]
        Rf = peak_Rfs[idx]
        ir2min = min(floor(Int, (1.75 * Rf / alatt)^2),
                     floor(Int, (40.0 / alatt - 1)^2))

        ZZon_pk = ZZon
        if ievol == 1
            z_pk = Huntian.peak_redshift(obs[1], obs[2], obs[3], peak.x, peak.y, peak.z, chi2z)
            ZZon_pk = 1.0 + z_pk
            z_pk > z_max && continue
        end

        result = Huntian.analyse_peak(pg, peak.ipp, alatt, ir2min, ZZon_pk, Rf, ct, shells;
                                         nbuff=nbuff, rmax2rs=cfg.rmax2rs)

        if result.RTHL > 0
            a_pk = 1.0 / ZZon_pk
            D_pk, _, _ = Huntian.Dlinear_ab(a_pk, growth_tables)   # 1st return is D (3rd is D/a)

            RTHL_phys = Float32(result.RTHL * alatt)
            Sbar_vel = result.Sbar .* D_pk

            Sbar2_vel = zeros(3)
            if ilpt >= 2
                Om_a = Huntian.Cosmology.omega_m_a(Omnr, cosmo.OL, a_pk)
                Sbar2_vel = result.Sbar2 .* (-3.0/7.0 * Om_a^(-1.0/143) * D_pk^2)   # -3/7 matches Fortran
            end

            if ioutshear >= 1
                sm = result.strain_mat
                push!(halos_ext, Huntian.ExtHaloRecord(
                    Float32(peak.x), Float32(peak.y), Float32(peak.z),
                    Float32(Sbar_vel[1]), Float32(Sbar_vel[2]), Float32(Sbar_vel[3]),
                    RTHL_phys,
                    Float32(Sbar2_vel[1]), Float32(Sbar2_vel[2]), Float32(Sbar2_vel[3]),
                    Float32(result.Fbarx),
                    Float32(result.e_v), Float32(result.p_v),
                    Float32(sm[1,1]), Float32(sm[2,2]), Float32(sm[3,3]),
                    Float32(sm[2,3]), Float32(sm[1,3]), Float32(sm[1,2]),
                    Float32(result.d2F),
                    Float32(result.zvir_half),
                    Float32(result.gradpk[1]), Float32(result.gradpk[2]), Float32(result.gradpk[3]),
                    Float32(result.gradpkf[1]), Float32(result.gradpkf[2]), Float32(result.gradpkf[3]),
                    Float32(Rf),
                    peak_FcRfs[idx],
                    peak_d2Rfs[idx],
                    Float32(result.gradpkrf[1]), Float32(result.gradpkrf[2]), Float32(result.gradpkrf[3])
                ))
            else
                push!(halos_basic, Huntian.HaloRecord(
                    Float32(peak.x), Float32(peak.y), Float32(peak.z),
                    Float32(Sbar_vel[1]), Float32(Sbar_vel[2]), Float32(Sbar_vel[3]),
                    RTHL_phys,
                    Float32(Sbar2_vel[1]), Float32(Sbar2_vel[2]), Float32(Sbar2_vel[3]),
                    Float32(result.Fbarx)
                ))
            end
        end
    end
end

function Huntian.run_multitile_mpi(cfg::Huntian.PipelineConfig;
        ntile::Int, seed::Integer=42, verbose::Bool=false,
        comm::MPI.Comm=MPI.COMM_WORLD, lowmem::Bool=false,
        available_gb::Float64=900.0)

    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Geometry
    nmesh  = cfg.n
    nbuff  = cfg.nbuff
    nsub   = nmesh - 2 * nbuff
    N      = nsub * ntile + 2 * nbuff
    cfg.periodic_cores && error("periodic_cores=true is not supported by the MPI path " *
        "(tiles are sliced from pencils without wrapping); use run_multitile_split")
    alatt  = cfg.boxsize / nmesh
    boxsize_full = N * alatt
    dcore_box = nsub * alatt

    # Phase 0: initialization (all ranks, no communication)
    Om_total = cfg.Omx + cfg.OmB
    cosmo = Huntian.CosmologyParams(Om_total, cfg.OmB, cfg.Omvac,
                                       cfg.h, 0.965, 0.808)
    growth_tables = Huntian.Dlinear_tables(cosmo)

    ct_array, ct_params = Huntian.read_homeltab(cfg.tabfile)
    ct = Huntian.CollapseTableInterp(ct_array, ct_params)

    filters = Huntian.read_filterbank(cfg.filterfile)
    sort!(filters; by=f -> -f[3])

    z_out  = cfg.z_out
    a_out  = 1.0 / (1.0 + z_out)
    ZZon   = 1.0 + z_out
    fcrit  = Float32(Huntian.fsc_of_z(z_out, ct))
    _, _, D_out = Huntian.Dlinear_ab(a_out, growth_tables)

    Rfclmax = filters[1][3]
    nhunt  = min(nbuff - 1, floor(Int, Rfclmax * 1.75 / alatt))
    shells = Huntian.precompute_shells(nhunt)

    Omnr    = cosmo.Om
    vTHvir0 = 100.0 * cosmo.h * sqrt(Omnr)
    ioutshear = cfg.ioutshear
    wsmooth   = cfg.wsmooth
    ilpt      = cfg.ilpt

    cfg.NonGauss != 0 && error("run_multitile_mpi: NonGauss=$(cfg.NonGauss) not yet supported (only 0)")

    # Lightcone mode
    ievol = cfg.ievol
    obs = (cfg.cenx, cfg.ceny, cfg.cenz)
    z_max = cfg.z_max
    chi2z = ievol == 1 ? Huntian.build_chi_to_z(cosmo; z_max=z_max + 1.0) : nothing

    my_tiles = _assign_tiles(ntile, nranks, rank)

    # Lightcone: drop tiles whose nearest point lies beyond chi(z_max)
    chi_max = 0.0
    if ievol == 1
        chi_max = Huntian.chi(z_max, cosmo)
        half = dcore_box / 2.0
        filter!(my_tiles) do tid
            it, jt, kt = tid
            xbx, ybx, zbx = Huntian.tile_center(it, jt, kt, ntile, dcore_box)
            dx = max(abs(xbx - obs[1]) - half, 0.0)
            dy = max(abs(ybx - obs[2]) - half, 0.0)
            dz = max(abs(zbx - obs[3]) - half, 0.0)
            sqrt(dx^2 + dy^2 + dz^2) <= chi_max
        end
    end

    rank == 0 && verbose && @info "Phase 0: N=$N, ntile=$ntile, nranks=$nranks, tiles/rank=$(length(my_tiles))$(ievol == 1 ? ", lightcone mode" : "")$(lowmem ? ", lowmem" : "")"

    # PencilFFT setup: in-place RFFT! in Float32 shares one buffer between the real and
    # complex views, avoiding separate input/output arrays (~310 GB/node at 6144³).
    nthreads = Threads.nthreads()
    FFTW.set_num_threads(nthreads)
    rank == 0 && verbose && @info "FFTW threads: $nthreads"
    proc_dims = _factor_procs(nranks)
    _plan_tmp = PencilFFTPlan((N, N, N), RFFT(), proc_dims, comm)
    _noise_tmp = PencilFFTs.allocate_input(_plan_tmp)
    _pen = PencilArrays.pencil(_noise_tmp)
    noise_pencil = PencilArray{Float32}(undef, _pen)
    _plan_tmp = nothing; _noise_tmp = nothing
    plan = PencilFFTPlan(noise_pencil, RFFT!())
    noise_pencil = nothing
    # A is a ManyPencilArrayRFFT!: first(A) is the real view, last(A) the complex k-space
    # view, sharing one buffer.
    A = PencilFFTs.allocate_input(plan)

    # Phase 1a: distributed white noise (Threefry counter-based RNG)
    pk = Huntian.load_pk(cfg.pkfile)

    pen = PencilArrays.pencil(first(A))
    ranges = PencilArrays.range_local(pen)
    Huntian.fill_noise_threefry_region!(parent(first(A)), ranges, N, seed)

    rank == 0 && verbose && @info "Phase 1a: distributed noise generated (Threefry, Float32)"

    # Phase 1b: forward FFT + P(k) convolution
    plan * A
    # last(A) now contains delta_k in the final pencil configuration
    _distributed_convolve_pk!(last(A), pk, N, boxsize_full)

    # Separate copy: last(A) is overwritten by every inverse FFT below
    delta_k_saved = similar(last(A))
    delta_k_saved .= last(A)

    rank == 0 && verbose && @info "Phase 1b: forward FFT + P(k) done"

    # Phase 1 field extraction (standard vs lowmem). Every inverse FFT is in place:
    # write the kernel into last(A), plan \ A, read first(A).

    delta_tiles = nothing
    psi_tiles = nothing
    psi2_tiles = nothing
    lapd_tiles = nothing
    src2_k_pencil = nothing

    if !lowmem
        # Standard: extract all tile fields now
        last(A) .= delta_k_saved
        plan \ A
        delta_tiles = _extract_my_tiles(first(A), my_tiles, ntile, nsub, nmesh,
                                        Float32, comm)

        psi_tiles = Dict{Int, Dict{NTuple{3,Int}, Array{Float32,3}}}()
        for dim in 1:3
            _apply_1lpt_kernel!(last(A), delta_k_saved, dim, N, boxsize_full)
            plan \ A
            psi_tiles[dim] = _extract_my_tiles(first(A), my_tiles, ntile, nsub, nmesh,
                                               Float32, comm)
        end

        rank == 0 && verbose && @info "Phase 1d: 1LPT done"

        if ilpt >= 2
            # src2 = (δ² - Σ phi_ii²) / 2 - Σ_{i<j} phi_ij², from the trace identity
            # Σ_{i<j} phi_ii*phi_jj = ((Σ phi_ii)² - Σ phi_ii²)/2 with Σ phi_ii = -δ.
            # Avoids holding two full phi arrays at once.
            src2_pencil = PencilArray{Float32}(undef, PencilArrays.pencil(first(A)))
            parent(src2_pencil) .= 0.0f0

            # (Σ phi_ii)² / 2 = δ² / 2, with δ's k=0 mode zeroed as in phi_ij
            _apply_trace_kernel!(last(A), delta_k_saved, N)
            plan \ A
            parent(src2_pencil) .+= parent(first(A)) .^ 2 .* 0.5f0

            for d in 1:3
                _apply_phi_ij_kernel!(last(A), delta_k_saved, d, d, N, boxsize_full)
                plan \ A
                parent(src2_pencil) .-= parent(first(A)) .^ 2 .* 0.5f0
            end

            for (di, dj) in ((1,2), (1,3), (2,3))
                _apply_phi_ij_kernel!(last(A), delta_k_saved, di, dj, N, boxsize_full)
                plan \ A
                parent(src2_pencil) .-= parent(first(A)) .^ 2
            end

            parent(first(A)) .= parent(src2_pencil)
            src2_pencil = nothing; GC.gc()
            plan * A
            src2_k_local = similar(last(A))
            src2_k_local .= last(A)

            psi2_tiles = Dict{Int, Dict{NTuple{3,Int}, Array{Float32,3}}}()
            for dim in 1:3
                _apply_2lpt_kernel!(last(A), src2_k_local, dim, N, boxsize_full)
                plan \ A
                psi2_tiles[dim] = _extract_my_tiles(first(A), my_tiles, ntile, nsub, nmesh,
                                                    Float32, comm)
            end
            src2_k_local = nothing; GC.gc()

            rank == 0 && verbose && @info "Phase 1e: 2LPT done (trace identity src2)"
        end

        if ioutshear >= 1
            _apply_laplacian_kernel!(last(A), delta_k_saved, N, boxsize_full)
            plan \ A
            lapd_tiles = _extract_my_tiles(first(A), my_tiles, ntile, nsub, nmesh,
                                           Float32, comm)
            scale = Float32(1.0 / Float64(N)^3)
            for (_, arr) in lapd_tiles
                arr .*= scale
            end

            rank == 0 && verbose && @info "Phase 1f: Laplacian done"
        end

    else
        # Lowmem: compute only src2_k; tile extraction is deferred to the batched Phase 3-4

        if ilpt >= 2
            src2_pencil = PencilArray{Float32}(undef, PencilArrays.pencil(first(A)))
            parent(src2_pencil) .= 0.0f0

            # Trace identity, as in the standard path
            _apply_trace_kernel!(last(A), delta_k_saved, N)
            plan \ A
            parent(src2_pencil) .+= parent(first(A)) .^ 2 .* 0.5f0

            for d in 1:3
                _apply_phi_ij_kernel!(last(A), delta_k_saved, d, d, N, boxsize_full)
                plan \ A
                parent(src2_pencil) .-= parent(first(A)) .^ 2 .* 0.5f0
            end

            for (di, dj) in ((1,2), (1,3), (2,3))
                _apply_phi_ij_kernel!(last(A), delta_k_saved, di, dj, N, boxsize_full)
                plan \ A
                parent(src2_pencil) .-= parent(first(A)) .^ 2
            end

            parent(first(A)) .= parent(src2_pencil)
            src2_pencil = nothing; GC.gc()
            plan * A
            src2_k_pencil = similar(last(A))
            src2_k_pencil .= last(A)

            rank == 0 && verbose && @info "Phase 1e: src2_k computed (lowmem, trace identity)"
        end
    end

    # Phase 2: multi-scale peak finding (both modes)

    tile_masks    = Dict(tid => zeros(Int8, nmesh, nmesh, nmesh) for tid in my_tiles)
    tile_peaks    = Dict(tid => Huntian.PeakCandidate[] for tid in my_tiles)
    tile_peak_Rf  = Dict(tid => Float64[] for tid in my_tiles)
    tile_peak_FcRf = Dict(tid => Float32[] for tid in my_tiles)
    tile_peak_d2Rf = Dict(tid => Float32[] for tid in my_tiles)

    for ic in 1:length(filters)
        Rf = filters[ic][3]

        _apply_window_kernel!(last(A), delta_k_saved, N, boxsize_full, Rf, wsmooth)
        plan \ A
        delta_s_tiles = _extract_my_tiles(first(A), my_tiles, ntile, nsub, nmesh,
                                          Float32, comm)

        lapd_s_tiles = nothing
        if ioutshear >= 1
            _apply_window_kernel!(last(A), delta_k_saved, N, boxsize_full, Rf, 3)
            plan \ A
            lapd_s_tiles = _extract_my_tiles(first(A), my_tiles, ntile, nsub, nmesh,
                                             Float32, comm)
        end

        filter_peak_count = 0
        for tid in my_tiles
            it, jt, kt = tid
            delta_s_tile = delta_s_tiles[tid]
            xbx, ybx, zbx = Huntian.tile_center(it, jt, kt, ntile, dcore_box)

            # Constant fcrit = fsc_of_z(z_out), as in Fortran; the per-peak lightcone
            # redshift enters only in the shell analysis.
            fcrit_tile = fcrit

            new_peaks = Huntian.find_peaks(delta_s_tile, tile_masks[tid],
                                              xbx, ybx, zbx, alatt, nbuff, fcrit_tile, Rf)

            append!(tile_peaks[tid], new_peaks)
            append!(tile_peak_Rf[tid], fill(Rf, length(new_peaks)))

            if !isempty(new_peaks)
                lapd_s_tile = lapd_s_tiles !== nothing ? lapd_s_tiles[tid] : nothing
                for pk in new_peaks
                    i, j, k = _ipp_to_ijk(pk.ipp, nmesh)
                    push!(tile_peak_FcRf[tid], delta_s_tile[i, j, k])
                    push!(tile_peak_d2Rf[tid],
                          lapd_s_tile !== nothing ? lapd_s_tile[i, j, k] : 0.0f0)
                end
            end
            filter_peak_count += length(new_peaks)
        end

        rank == 0 && verbose && @info "  Filter $ic: Rf=$(round(Rf;digits=3)), peaks=$filter_peak_count"
    end

    if !lowmem
        delta_k_saved = nothing; GC.gc()
    end

    total_peaks = sum(length(tile_peaks[tid]) for tid in my_tiles; init=0)
    rank == 0 && verbose && @info "Phase 2 done: $total_peaks peaks on rank $rank"

    # Phase 3-4: shell analysis

    halos_basic = Huntian.HaloRecord[]
    halos_ext   = Huntian.ExtHaloRecord[]

    if !lowmem
        for tid in my_tiles
            _analyse_tile_shells!(halos_basic, halos_ext,
                tid, tile_peaks[tid], tile_peak_Rf[tid],
                tile_peak_FcRf[tid], tile_peak_d2Rf[tid], tile_masks[tid],
                delta_tiles[tid],
                psi_tiles[1][tid], psi_tiles[2][tid], psi_tiles[3][tid],
                psi2_tiles !== nothing ? psi2_tiles[1][tid] : nothing,
                psi2_tiles !== nothing ? psi2_tiles[2][tid] : nothing,
                psi2_tiles !== nothing ? psi2_tiles[3][tid] : nothing,
                lapd_tiles !== nothing ? lapd_tiles[tid] : nothing,
                nmesh, nbuff, alatt, ntile, dcore_box,
                obs, ievol, z_max, chi2z, growth_tables, ZZon,
                Omnr, cosmo, ct, shells, cfg, ioutshear, ilpt)
        end
    else
        # Lowmem: batched tile extraction + shell analysis. The active tile list must be
        # identical on all ranks (every rank takes part in each batch's extraction).
        all_active_tiles = NTuple{3,Int}[]
        half_lc = dcore_box / 2.0
        for kt in 1:ntile, jt in 1:ntile, it in 1:ntile
            tid = (it, jt, kt)
            if ievol == 1
                xbx, ybx, zbx = Huntian.tile_center(it, jt, kt, ntile, dcore_box)
                dx = max(abs(xbx - obs[1]) - half_lc, 0.0)
                dy = max(abs(ybx - obs[2]) - half_lc, 0.0)
                dz = max(abs(zbx - obs[3]) - half_lc, 0.0)
                sqrt(dx^2 + dy^2 + dz^2) <= chi_max || continue
            end
            push!(all_active_tiles, tid)
        end

        # Batch size from the memory budget. Peak during an IFFT:
        # delta_k + src2_k + field_k + field_real + (n_fields-1) batch tiles
        # = (n_stored + 2) × dk_bytes + (n_fields-1) × B × tile_bytes
        local_dk_bytes = sizeof(parent(delta_k_saved))
        tile_field_bytes = nmesh^3 * sizeof(Float32)
        n_stored = src2_k_pencil !== nothing ? 2 : 1
        n_tile_fields = 1 + 3 + (ilpt >= 2 ? 3 : 0) + (ioutshear >= 1 ? 1 : 0)
        overhead_bytes = (n_stored + 2) * local_dk_bytes
        avail_for_tiles = available_gb * 1e9 - overhead_bytes
        tiles_per_batch = max(1, floor(Int, avail_for_tiles / ((n_tile_fields - 1) * tile_field_bytes)))

        my_tile_set = Set(my_tiles)
        n_batches = cld(length(all_active_tiles), tiles_per_batch)

        rank == 0 && verbose && @info "Phase 3-4: lowmem batched, $(length(all_active_tiles)) active tiles, batch_size=$tiles_per_batch, $n_batches batches"

        for bi in 1:n_batches
            batch_start = (bi - 1) * tiles_per_batch + 1
            batch_end = min(bi * tiles_per_batch, length(all_active_tiles))
            batch_global = all_active_tiles[batch_start:batch_end]
            my_batch = [t for t in batch_global if t in my_tile_set]

            rank == 0 && verbose && @info "  Batch $bi/$n_batches: $(length(batch_global)) tiles ($(length(my_batch)) local)"

            last(A) .= delta_k_saved
            plan \ A
            batch_delta = _extract_my_tiles(first(A), my_batch, ntile, nsub, nmesh,
                                            Float32, comm; global_tile_list=batch_global)

            batch_psi = Dict{Int, Dict{NTuple{3,Int}, Array{Float32,3}}}()
            for dim in 1:3
                _apply_1lpt_kernel!(last(A), delta_k_saved, dim, N, boxsize_full)
                plan \ A
                batch_psi[dim] = _extract_my_tiles(first(A), my_batch, ntile, nsub, nmesh,
                                                   Float32, comm; global_tile_list=batch_global)
            end

            batch_psi2 = Dict{Int, Dict{NTuple{3,Int}, Array{Float32,3}}}()
            if ilpt >= 2 && src2_k_pencil !== nothing
                for dim in 1:3
                    _apply_2lpt_kernel!(last(A), src2_k_pencil, dim, N, boxsize_full)
                    plan \ A
                    batch_psi2[dim] = _extract_my_tiles(first(A), my_batch, ntile, nsub, nmesh,
                                                        Float32, comm; global_tile_list=batch_global)
                end
            end

            batch_lapd = nothing
            if ioutshear >= 1
                _apply_laplacian_kernel!(last(A), delta_k_saved, N, boxsize_full)
                plan \ A
                batch_lapd = _extract_my_tiles(first(A), my_batch, ntile, nsub, nmesh,
                                               Float32, comm; global_tile_list=batch_global)
                scale = Float32(1.0 / Float64(N)^3)
                for (_, arr) in batch_lapd
                    arr .*= scale
                end
            end

            for tid in my_batch
                _analyse_tile_shells!(halos_basic, halos_ext,
                    tid, tile_peaks[tid], tile_peak_Rf[tid],
                    tile_peak_FcRf[tid], tile_peak_d2Rf[tid], tile_masks[tid],
                    batch_delta[tid],
                    batch_psi[1][tid], batch_psi[2][tid], batch_psi[3][tid],
                    haskey(batch_psi2, 1) ? batch_psi2[1][tid] : nothing,
                    haskey(batch_psi2, 2) ? batch_psi2[2][tid] : nothing,
                    haskey(batch_psi2, 3) ? batch_psi2[3][tid] : nothing,
                    batch_lapd !== nothing ? batch_lapd[tid] : nothing,
                    nmesh, nbuff, alatt, ntile, dcore_box,
                    obs, ievol, z_max, chi2z, growth_tables, ZZon,
                    Omnr, cosmo, ct, shells, cfg, ioutshear, ilpt)
            end

            batch_delta = nothing; batch_psi = nothing
            batch_psi2 = nothing; batch_lapd = nothing; GC.gc()
        end

        delta_k_saved = nothing; src2_k_pencil = nothing; GC.gc()
    end

    local_halos = ioutshear >= 1 ? halos_ext : halos_basic

    rank == 0 && verbose && @info "Phase 3-4 done: $(length(local_halos)) halos on rank $rank"

    # Phase 5: gather halos to rank 0 as flat Float32 records
    nfields = ioutshear >= 1 ? fieldcount(Huntian.ExtHaloRecord) : fieldcount(Huntian.HaloRecord)
    T_halo = ioutshear >= 1 ? Huntian.ExtHaloRecord : Huntian.HaloRecord

    local_data = Float32[]
    for h in local_halos
        for fn in fieldnames(typeof(h))
            push!(local_data, getfield(h, fn))
        end
    end

    local_count = Int32[length(local_data)]
    all_counts = MPI.Allgather(local_count, comm)
    all_counts_vec = vec(all_counts)

    if rank == 0
        total_floats = sum(all_counts_vec)
        recvbuf = Vector{Float32}(undef, total_floats)
        MPI.Gatherv!(local_data, MPI.VBuffer(recvbuf, all_counts_vec), 0, comm)

        all_halos = T_halo[]
        pos = 1
        while pos + nfields - 1 <= length(recvbuf)
            vals = ntuple(i -> recvbuf[pos + i - 1], nfields)
            push!(all_halos, T_halo(vals...))
            pos += nfields
        end

        verbose && @info "Phase 5: gathered $(length(all_halos)) halos on rank 0"
        return all_halos
    else
        MPI.Gatherv!(local_data, nothing, 0, comm)
        return T_halo[]
    end
end

# Install the mpi_buffer override at load time, not precompile time
function __init__()
    @eval Transpositions.mpi_buffer(buf::AbstractArray, off, len) =
        _mpi_buffer_large(buf, off, len)
end

end # module MPIExt
