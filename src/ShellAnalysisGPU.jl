module ShellAnalysisGPU

# CPU reference for the block-per-peak GPU shell analysis: the algorithm of
# RadialShell.analyse_peak, with lattice cells grouped into per-shell offset tables
# (one block per peak, threads gather the cells of a shell and block-reduce).

using StaticArrays

using ..RadialShell: PeakGrid, PeakResult, ShellCell, precompute_shells,
    get_ijk, no_collapse, kernel_strain, get_evals, normalize_strain,
    hRinteg, atab4, fsc_of_z, _zero_smat3, _zero_svec3, FOURPI
using ..CollapseTable: CollapseTableInterp, interpolate

export ShellTables, build_shell_tables, analyse_peak_gpu

"""
Precomputed shell offset tables grouped by radius.

Fields:
- `offsets_di`, `offsets_dj`, `offsets_dk`: flat arrays of cell offsets,
  grouped by shell. `offsets_di[shell_start[s]:shell_start[s]+shell_count[s]-1]`
  gives all di values for shell s.
- `shell_start`: 1-based start index into offsets for each shell
- `shell_count`: number of cells in each shell
- `shell_r2`: integer r² value for each shell
- `nshells`: total number of distinct shells
"""
struct ShellTables
    offsets_di::Vector{Int32}
    offsets_dj::Vector{Int32}
    offsets_dk::Vector{Int32}
    shell_start::Vector{Int}
    shell_count::Vector{Int}
    shell_r2::Vector{Int32}
    nshells::Int
end

"""
    build_shell_tables(rmax::Int) -> ShellTables

Build shell offset tables from `precompute_shells`: cells grouped by r², with
start/count arrays for direct indexing. Built once and reused for all tiles.
"""
function build_shell_tables(rmax::Int)
    cells = precompute_shells(rmax)

    di_all = Int32[]
    dj_all = Int32[]
    dk_all = Int32[]
    starts = Int[]
    counts = Int[]
    r2_vals = Int32[]

    i = 1
    while i <= length(cells)
        r2 = cells[i].r2
        push!(starts, length(di_all) + 1)
        push!(r2_vals, r2)
        count = 0
        while i <= length(cells) && cells[i].r2 == r2
            push!(di_all, cells[i].di)
            push!(dj_all, cells[i].dj)
            push!(dk_all, cells[i].dk)
            count += 1
            i += 1
        end
        push!(counts, count)
    end

    return ShellTables(di_all, dj_all, dk_all, starts, counts, r2_vals, length(starts))
end

"""
    analyse_peak_gpu(pg, ipp, alatt, ir2min, ZZon, Rfclvi, ct, stab;
                     nbuff=0, rmax2rs=0.0, ...) -> PeakResult

Same collapse criterion and outputs as `RadialShell.analyse_peak`, but gathering
from `ShellTables`; on GPU, peaks map to blocks and the cells of each shell to threads.
Does not compute the formation redshift (`zvir_half = 0`).
"""
function analyse_peak_gpu(pg::PeakGrid, ipp::Int, alatt::Float64, ir2min::Int,
                           ZZon::Float64, Rfclvi::Float64, ct::CollapseTableInterp,
                           stab::ShellTables;
                           nbuff::Int=0,
                           rmax2rs::Float64=0.0, fcrit_override=nothing,
                           fortran_compat::Bool=false)
    # `RadialShell.analyse_peak` finalises a shell only when the next cell has a
    # different r², so its last shell is never finalised: use `nshells - 1` to match.
    nshells_max = stab.nshells - 1
    if rmax2rs > 0.0
        f3p = fortran_compat ? Float32(4.0f0/3.0f0 * Float32(pi)) : 4.0/3.0 * pi
        r2_limit = (rmax2rs * Rfclvi / alatt)^2
        nshells_max = 0
        for s in 1:stab.nshells - 1
            stab.shell_r2[s] <= r2_limit || break
            nshells_max = s
        end
    end
    nshells_max < 2 && return no_collapse()

    hlatt = 1.0
    anor = 1.0 / hlatt^3
    aRnor = 3.0 * anor / alatt
    wnor = anor / FOURPI
    wRnor = 3.0 * wnor / alatt
    akk_tab = atab4()

    n1, n2, n3 = pg.nn
    n1xn2 = n1 * n2
    ipk = get_ijk(n1xn2, n1, ipp)

    # Skip peaks already masked by a larger halo.
    if rmax2rs > 0.0 && pg.mask !== nothing
        if pg.mask[ipk[1], ipk[2], ipk[3]] > floor(Int, Rfclvi / alatt * rmax2rs)
            return no_collapse()
        end
    end

    fcrit = fcrit_override !== nothing ? Float64(fcrit_override) :
            fsc_of_z(ZZon - 1.0, ct)

    ilpt2 = pg.eta2x !== nothing

    # Radial profile accumulation: shells are sequential, cells within a shell are
    # gathered in parallel on GPU (serially here).
    maxm = nshells_max + 2
    rad = Vector{Float64}(undef, maxm)
    Fbar = Vector{Float64}(undef, maxm)
    nshell_arr = Vector{Int}(undef, maxm)
    Sshell = [zeros(Float64, maxm) for _ in 1:3]
    S2shell = [zeros(Float64, maxm) for _ in 1:3]
    Gshell = [zeros(Float64, maxm) for _ in 1:3]
    Gshellf = [zeros(Float64, maxm) for _ in 1:3]
    SRshell = [[zeros(Float64, maxm) for _ in 1:3] for _ in 1:3]

    # Centre cell (r² = 0).
    m = 1
    rad[1] = 0.0
    Fbar[1] = pg.delta[ipk[1], ipk[2], ipk[3]]
    nshell_arr[1] = 1
    for L in 1:3
        eta_L = L == 1 ? pg.etax[ipk...] : L == 2 ? pg.etay[ipk...] : pg.etaz[ipk...]
        Sshell[L][1] = eta_L
        S2shell[L][1] = ilpt2 ? (L == 1 ? pg.eta2x[ipk...] : L == 2 ? pg.eta2y[ipk...] : pg.eta2z[ipk...]) : 0.0
    end

    dFbarp = Fbar[1]
    m0 = 1; ifcrit = 1; mupp = 1
    rupp_init = sqrt(Float64(ir2min)) + 2.0

    first_shell = stab.shell_r2[1] == 0 ? 2 : 1

    for s in first_shell:nshells_max
        r2 = stab.shell_r2[s]
        ncells = stab.shell_count[s]
        s0 = stab.shell_start[s]

        Fshell = 0.0
        nsh = 0
        local_S = zeros(Float64, 3)
        local_S2 = zeros(Float64, 3)
        local_G = zeros(Float64, 3)
        local_Gf = zeros(Float64, 3)
        local_SR = zeros(Float64, 3, 3)

        for c in 0:ncells-1
            di = stab.offsets_di[s0 + c]
            dj = stab.offsets_dj[s0 + c]
            dk = stab.offsets_dk[s0 + c]

            iv1 = ipk[1] + di
            iv2 = ipk[2] + dj
            iv3 = ipk[3] + dk

            (iv1 < 1 || iv1 > n1 || iv2 < 1 || iv2 > n2 || iv3 < 1 || iv3 > n3) && continue

            nsh += 1
            delta_val = pg.delta[iv1, iv2, iv3]
            Fshell += delta_val

            for L in 1:3
                eta_L = L == 1 ? pg.etax[iv1, iv2, iv3] : L == 2 ? pg.etay[iv1, iv2, iv3] : pg.etaz[iv1, iv2, iv3]
                local_S[L] += eta_L
                if ilpt2
                    local_S2[L] += (L == 1 ? pg.eta2x[iv1, iv2, iv3] : L == 2 ? pg.eta2y[iv1, iv2, iv3] : pg.eta2z[iv1, iv2, iv3])
                end
                ix_L = L == 1 ? di : L == 2 ? dj : dk
                local_G[L] += delta_val * ix_L
                local_Gf[L] += pg.delta[iv3, iv1, iv2] * ix_L
                for K in 1:3
                    ix_K = K == 1 ? di : K == 2 ? dj : dk
                    local_SR[L, K] += eta_L * ix_K
                end
            end
        end

        m += 1
        rad[m] = sqrt(Float64(r2))
        nshell_arr[m] = nsh
        dFbar = nsh > 0 ? Fshell / nsh : 0.0
        rad3p = rad[m-1]^3
        rad3 = rad[m]^3
        Fbar[m] = (rad3p * Fbar[m-1] + 0.5 * (dFbarp + dFbar) * (rad3 - rad3p)) / rad3
        dFbarp = dFbar

        if ifcrit == 1
            m0 = m
            if r2 > ir2min
                rupp_init = rad[m0] + 2.0
                ifcrit = 0
                mupp = m
            end
        else
            if rad[m] < rupp_init
                mupp = m
            end
        end

        # Gradient and strain sums are divided by the shell radius.
        for L in 1:3
            Sshell[L][m] = local_S[L]
            S2shell[L][m] = local_S2[L]
            Gshell[L][m] = local_G[L] / rad[m]
            Gshellf[L][m] = local_Gf[L] / rad[m]
            for K in 1:3
                SRshell[L][K][m] = local_SR[L, K] / rad[m]
            end
        end
    end

    # From here on the code follows RadialShell.analyse_peak on the per-shell profiles.

    # Gradient at the filter scale; mrf tracking mirrors analyse_peak's per-cell loop.
    rmrf = 10.0; mrf = 1
    rfclvi_r2 = (Rfclvi / alatt)^2
    mrfi = 1
    first_r2 = stab.shell_r2[1] == 0 ? (stab.nshells >= 2 ? stab.shell_r2[2] : Int32(0)) : stab.shell_r2[1]
    for s in (stab.shell_r2[1] == 0 ? 2 : 1):min(nshells_max, stab.nshells)
        r2 = stab.shell_r2[s]
        ncells_s = stab.shell_count[s]
        for c in 1:ncells_s
            dist = abs(Float64(r2) - rfclvi_r2)
            if dist < rmrf
                mrf = mrfi
                rmrf = dist
            end
        end
        if s < min(nshells_max, stab.nshells)
            next_r2 = stab.shell_r2[s+1]
            if next_r2 != r2
                mrfi += 1
            end
        end
    end
    rlow_rf = max(rad[mrf] - 2.0, 0.0)
    mlow_rf = mrf
    if mrf > 1
        for m1 in mrf-1:-1:1
            rad[m1] > rlow_rf && (mlow_rf = m1)
        end
    end
    rupp_rf = rad[mrf] + 2.0
    mupp_rf = mrf
    for m1 in mrf+1:m
        rad[m1] < rupp_rf && (mupp_rf = m1)
    end
    _, gradpkrf, _ = kernel_strain(rad, Gshell, Gshellf, SRshell,
                                    mrf, mlow_rf, mupp_rf, akk_tab,
                                    wRnor, aRnor, 1.0, 1.0)

    # Find the shell m0 where Fbar crosses fcrit.
    if Fbar[m0] >= fcrit
        found = false
        for mm in m0:m
            if Fbar[mm] < fcrit
                m0 = mm
                found = true
                break
            end
        end
        if !found
            mupp = m
        end
        if found
            rupp_m0 = rad[m0] + 2.0
            mupp = m0
            for mm in m0+1:m
                rad[mm] < rupp_m0 && (mupp = mm)
            end
        end
    else
        mstart = max(m0 - 1, 1)
        found = false
        for mp in mstart:-1:1
            if Fbar[mp] >= fcrit
                found = true
                break
            end
            m0 = mp
        end
        if !found
            return no_collapse()
        end
        rupp_m0 = rad[m0] + 2.0
        mupp_new = m0
        for mm in m0+1:m
            rad[mm] < rupp_m0 && (mupp_new = mm)
        end
        mupp = mupp_new
    end

    # Step inward from m0 until a shell has virialized by ZZon.
    mlow = m0
    rlow = max(rad[m0] - 2.0, 0.0)
    if m0 > 1
        for m1 in m0-1:-1:1
            rad[m1] > rlow && (mlow = m1)
        end
    end

    zvir1 = -1.0; zvir1p = -1.0
    Ebar_last = _zero_smat3
    gradpk_last = _zero_svec3; gradpkf_last = _zero_svec3
    Frhoh = 0.0; Frho_val = 0.0; e_v = 0.0; p_v = 0.0
    strain_mat = _zero_smat3
    gradpk_prev = _zero_svec3; gradpkf_prev = _zero_svec3
    Frhpk = 0.0; Fnupk = 0.0; Fevpk = 0.0; Fpvpk = 0.0
    collapsed = false

    Ebar_m0, gradpk_m0, gradpkf_m0 = kernel_strain(rad, Gshell, Gshellf, SRshell,
                                                     m0, mlow, mupp, akk_tab,
                                                     wRnor, aRnor, 1.0, 1.0)
    Lam_m0, _, iflag_m0 = get_evals(Ebar_m0)
    Frho_m0 = Lam_m0[1] + Lam_m0[2] + Lam_m0[3]
    e_v_m0 = Frho_m0 > 0 ? 0.5 * (Lam_m0[3] - Lam_m0[1]) / Frho_m0 : 0.0
    p_v_m0 = Frho_m0 > 0 ? 0.5 * (Lam_m0[3] + Lam_m0[1] - 2.0 * Lam_m0[2]) / Frho_m0 : 0.0
    Frhoh_m0 = Frho_m0
    Frho_m0 = Fbar[m0]
    # Fortran evaluates zvir here with an uninitialized Frhoc, effectively -1; replicated.
    zvir1p_m0 = -1.0

    Frhoh = Frhoh_m0; Frho_val = Frho_m0; e_v = e_v_m0; p_v = p_v_m0
    Ebar_last = Ebar_m0; gradpk_last = gradpk_m0; gradpkf_last = gradpkf_m0
    strain_mat = Ebar_m0; gradpk_prev = gradpk_m0; gradpkf_prev = gradpkf_m0
    Frhpk = Frhoh_m0; Fnupk = Frho_m0; Fevpk = e_v_m0; Fpvpk = p_v_m0

    if zvir1p_m0 >= ZZon
        zvir1p = zvir1p_m0
        collapsed = true
    else
        zvir1 = zvir1p_m0

        if m0 == 1
            return no_collapse()
        end

        for mp in m0-1:-1:1
            rupp_mp = rad[mp] + 2.0
            rlow_mp = max(rad[mp] - 2.0, 0.0)

            if rad[mupp] > rupp_mp
                mupp_new = mp
                for m1 in mp+1:mupp
                    rad[m1] < rupp_mp && (mupp_new = m1)
                end
                mupp = mupp_new
            end
            if mlow > 1 && rad[mlow-1] > rlow_mp
                mlownew = mlow
                for m1 in mlow-1:-1:1
                    rad[m1] > rlow_mp && (mlownew = m1)
                end
                mlow = mlownew
            end

            Ebar_mp, gradpk_mp, gradpkf_mp = kernel_strain(rad, Gshell, Gshellf, SRshell,
                                                             mp, mlow, mupp, akk_tab,
                                                             wRnor, aRnor, 1.0, 1.0)
            Lam_mp, _, iflag_mp = get_evals(Ebar_mp)
            Frho_mp = Lam_mp[1] + Lam_mp[2] + Lam_mp[3]
            e_v_mp = Frho_mp > 0 ? 0.5 * (Lam_mp[3] - Lam_mp[1]) / Frho_mp : 0.0
            p_v_mp = Frho_mp > 0 ? 0.5 * (Lam_mp[3] + Lam_mp[1] - 2.0 * Lam_mp[2]) / Frho_mp : 0.0

            Frhoh_mp = Frho_mp
            Frho_mp = Fbar[mp]

            zvir1p_mp = -1.0
            if iflag_mp == 0 && Frho_mp > 0.0
                poe_mp = e_v_mp < 1e-5 ? 0.0 : p_v_mp / e_v_mp
                zvir1p_mp = interpolate(ct, log10(Frho_mp), e_v_mp, poe_mp)
            end

            if zvir1p_mp >= ZZon
                m0 = mp + 1
                zvir1p = zvir1p_mp
                Ebar_last = Ebar_mp; gradpk_last = gradpk_mp; gradpkf_last = gradpkf_mp
                Frhoh = Frhoh_mp; Frho_val = Frho_mp; e_v = e_v_mp; p_v = p_v_mp
                collapsed = true
                break
            end

            zvir1 = zvir1p_mp
            Frhpk = Frhoh_mp; Fnupk = Frho_mp; Fevpk = e_v_mp; Fpvpk = p_v_mp
            strain_mat = Ebar_mp; gradpk_prev = gradpk_mp; gradpkf_prev = gradpkf_mp
            Frhoh = Frhoh_mp; Frho_val = Frho_mp; e_v = e_v_mp; p_v = p_v_mp
            Ebar_last = Ebar_mp; gradpk_last = gradpk_mp; gradpkf_last = gradpkf_mp
        end
    end

    !collapsed && return no_collapse()

    # Interpolate in r³ to the radius where zvir = ZZon.
    dZvir = zvir1p - zvir1
    RTHL = 0.0
    Fbarx = 0.0
    frac = 0.0
    if zvir1 > 0.0 && dZvir != 0.0
        RTHL3 = rad[m0-1]^3 + (rad[m0]^3 - rad[m0-1]^3) * (zvir1p - ZZon) / dZvir
        RTHL = RTHL3^(1.0/3.0)
    else
        RTHL = rad[m0-1]
    end

    RTHL <= 0 && return no_collapse()

    RTHL3 = RTHL^3
    RTHL5 = RTHL3 * RTHL * RTHL
    rad3p = rad[m0-1]^3
    rad3 = rad[m0]^3
    drad3 = rad3 - rad3p
    dFbar = Fbar[m0] - Fbar[m0-1]

    if zvir1 > 0.0 && drad3 != 0.0
        frac = (RTHL3 - rad3p) / drad3
        Fbarx = Fbar[m0-1] + frac * dFbar
        Frhpk = Frhoh + frac * (Frhpk - Frhoh)
        Fnupk = Frho_val + frac * (Fnupk - Frho_val)
        Fevpk = e_v + frac * (Fevpk - e_v)
        Fpvpk = p_v + frac * (Fpvpk - p_v)
        strain_mat = Ebar_last + frac * (strain_mat - Ebar_last)
        gradpk_final = gradpk_last + frac * (gradpk_prev - gradpk_last)
        gradpkf_final = gradpkf_last + frac * (gradpkf_prev - gradpkf_last)
    else
        Fbarx = Fbar[m0-1]
        Fnupk = Frho_val
        Fevpk = e_v
        Fpvpk = p_v
        strain_mat = Ebar_last
        gradpk_final = gradpk_prev
        gradpkf_final = gradpkf_prev
    end

    strain_norm = normalize_strain(strain_mat, Fbarx)
    Lam_f, _, _ = get_evals(strain_norm)
    Frhoc = Lam_f[1] + Lam_f[2] + Lam_f[3]
    e_v_f = Frhoc > 0 ? 0.5 * (Lam_f[3] - Lam_f[1]) / Frhoc : 0.0
    p_v_f = Frhoc > 0 ? 0.5 * (Lam_f[3] + Lam_f[1] - 2.0 * Lam_f[2]) / Frhoc : 0.0

    strain_final = Frhoh != 0.0 ? Ebar_last * (Fbarx / Frhoh) : Ebar_last

    # Mean displacement over collapsed shells.
    Sbar_m = @MVector zeros(3)
    Sbar2_m = @MVector zeros(3)
    nSbar = 0
    for m1 in 1:m0-1
        if nshell_arr[m1] > 0
            for L in 1:3
                Sbar_m[L] += Sshell[L][m1]
                Sbar2_m[L] += S2shell[L][m1]
            end
            nSbar += nshell_arr[m1]
        end
    end
    if nSbar > 0
        Sbar_m ./= nSbar
        Sbar2_m ./= nSbar
    end

    Srb = 0.0
    rad5p = 0.0
    if m0 > 2
        for mp in 2:m0-1
            rad5 = rad[mp]^5
            Srb += 0.5 * (Fbar[mp-1] + Fbar[mp]) * (rad5 - rad5p)
            rad5p = rad5
        end
    end
    if zvir1 > 0.0 && dZvir != 0.0
        Srb += 0.5 * (Fbar[m0-1] + Fbarx) * (RTHL5 - rad5p)
    end
    if Fbarx * RTHL5 > 0.0
        Srb /= (Fbarx * RTHL5)
    end

    # d2F: mean Laplacian over cells within RTHL.
    d2F = 0.0
    if pg.lapd !== nothing
        nd2 = 0
        for s in 1:stab.nshells
            sqrt(Float64(stab.shell_r2[s])) > RTHL && break
            for c in 0:stab.shell_count[s]-1
                iv1 = ipk[1] + stab.offsets_di[stab.shell_start[s]+c]
                iv2 = ipk[2] + stab.offsets_dj[stab.shell_start[s]+c]
                iv3 = ipk[3] + stab.offsets_dk[stab.shell_start[s]+c]
                (iv1 < 1 || iv1 > n1 || iv2 < 1 || iv2 > n2 || iv3 < 1 || iv3 > n3) && continue
                d2F += pg.lapd[iv1, iv2, iv3]
                nd2 += 1
            end
        end
        nd2 > 0 && (d2F /= nd2)
    end

    if pg.mask !== nothing
        for s in 1:stab.nshells
            sqrt(Float64(stab.shell_r2[s])) > RTHL && break
            for c in 0:stab.shell_count[s]-1
                iv1 = ipk[1] + stab.offsets_di[stab.shell_start[s]+c]
                iv2 = ipk[2] + stab.offsets_dj[stab.shell_start[s]+c]
                iv3 = ipk[3] + stab.offsets_dk[stab.shell_start[s]+c]
                (iv1 < 1 || iv1 > n1 || iv2 < 1 || iv2 > n2 || iv3 < 1 || iv3 > n3) && continue
                if iv1 > nbuff && iv1 <= n1 - nbuff &&
                   iv2 > nbuff && iv2 <= n2 - nbuff &&
                   iv3 > nbuff && iv3 <= n3 - nbuff
                    pg.mask[iv1, iv2, iv3] = Int8(1)
                end
            end
        end
    end

    zvir_half = 0.0  # formation redshift not computed here

    return PeakResult(RTHL, Srb, SVector{3}(Sbar_m), SVector{3}(Sbar2_m),
                      strain_final, gradpk_final, gradpkrf, Fbarx, e_v_f, p_v_f,
                      zvir_half, gradpkf_final, d2F)
end

end # module ShellAnalysisGPU
