#!/usr/bin/env julia
# Full-sky abundance matching to Tinker08 (as Websky, Stein+2020 §4.2): ONE table from the N(>M|z) counts of all
# 8 octants against the full-sky Tinker volume (fsky = 1), with the fractional correction frozen above the mass
# where fewer than `tail_N` halos remain in a z bin (see `build_abundance_table`).
#
#   julia --project=pipeline pipeline/abundance_match.jl table <table.txt> <tail_N> <z_max> <cat_oct000> ... <cat_oct111>
#       (catalog file names must contain "octZYX"; the observer comes from those bits)
#   julia --project=pipeline pipeline/abundance_match.jl apply <table.txt> <in.h5> <out.h5> <ZYX> [compact.h5]
#       the output carries the raw catalog's provenance plus an "am" stage; with a 5th argument also a quantised
#       compact catalog with the same provenance
using Huntian, Printf, HDF5

const COSMO = CosmologyParams(0.31, 0.049, 0.69, 0.68, 0.965, 0.81)   # first argument is Om TOTAL
const RHO_M = 2.775e11 * 0.31
const NMBINS, NZBINS, MMIN, MMAX = 10000, 46, 5e11, 1e16

# octZYX bit = 1 -> observer at +2618 Mpc/h on that axis
obs_of(zyx) = (zyx[3] == '1' ? 2618.0 : -2618.0, zyx[2] == '1' ? 2618.0 : -2618.0, zyx[1] == '1' ? 2618.0 : -2618.0)
octof(path) = (m = match(r"oct([01]{3})", basename(path)); m === nothing && error("no octZYX in $path"); m.captures[1])

NgtM(hs, M0) = count(h -> (4π / 3) * RHO_M * Float64(h.RTHL)^3 > M0, hs)
report(label, hs) = @printf("%-12s N(>1.7e12)=%.4e  N(>1e14)=%.4e  N(>1e15)=%d  N(>2e15)=%d\n", label,
                            NgtM(hs, 1.7e12), NgtM(hs, 1e14), NgtM(hs, 1e15), NgtM(hs, 2e15))

function make_table(out, tail_N, z_max, cats)
    length(cats) == 8 || error("need the 8 octant catalogs, got $(length(cats))")
    Medge, zedge = am_grid(; nMbins=NMBINS, z_min=0.0, z_max=z_max, nzbins=NZBINS, Mmin=MMIN, Mmax=MMAX)
    Npp = zeros(NMBINS, NZBINS)
    for c in cats
        oct = octof(c); @info "counting" c oct obs = obs_of(oct)
        halos, _, _ = read_halo_catalog(c)
        am_counts!(Npp, halos, COSMO; obs=obs_of(oct), Medge, zedge)
        halos = nothing; GC.gc()
    end
    @printf("full-sky counts: %d halos in [%.1e, %.1e), z<%.2f\n", sum(Npp), MMIN, MMAX, z_max)
    pk = Huntian.PowerSpectrum.load_pk(joinpath(@__DIR__, "..", "data", "pk_websky_RAW_unnormalized.dat"))
    t = build_abundance_table(Npp, COSMO, pk; z_min=0.0, z_max=z_max, Mmin=MMIN, Mmax=MMAX,
                              hmf=:tinker, fsky=1.0, tail_N=tail_N, verbose=true)
    save_abundance_table(out, t)
    @info "wrote table" out tail_N
end

function apply_table(tpath, inp, out, zyx, compact=nothing)
    halos, meta, prov = read_halo_catalog(inp)
    add_stage!(prov, "am"; scripts=[@__FILE__], inputs=Dict("table" => tpath, "raw_catalog" => inp),
               notes=Dict("octant" => zyx, "method" => "full-sky Tinker08 abundance matching (one table, tail_N in table header)"))
    t = load_abundance_table(tpath)
    report("RAW", halos)
    am = abundance_match(halos, t, COSMO; obs=obs_of(zyx))
    halos = nothing
    report("AM fullsky", am)
    prov["run.n_halos"] = string(length(am))
    prov["run.product"] = "abundance-matched catalog (Eulerian positions, km/s velocities, AM masses)"
    write_halo_catalog(out, am; z_out=meta["z_out"], provenance=prov,
                       attributes=Dict("observer" => meta["observer"], "stage" => "abundance-matched"))
    @info "wrote" out
    if compact !== nothing
        write_compact_catalog(compact, am, prov)
        @info "wrote compact catalog" compact size_GB = round(filesize(compact) / 1e9; digits=2)
    end
end

mode = ARGS[1]
if mode == "table"
    make_table(ARGS[2], parse(Float64, ARGS[3]), parse(Float64, ARGS[4]), ARGS[5:end])
elseif mode == "apply"
    apply_table(ARGS[2], ARGS[3], ARGS[4], ARGS[5], length(ARGS) >= 6 ? ARGS[6] : nothing)
else
    error("mode must be table or apply")
end
