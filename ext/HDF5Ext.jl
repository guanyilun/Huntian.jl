# HDF5 extension: the halo catalog format (lossless) and the compact (quantised) catalog.
module HDF5Ext

using Huntian
using HDF5

import Huntian: write_halo_catalog, read_halo_catalog, read_halo_columns, write_compact_catalog, read_compact_catalog
import TOML

import Huntian.Catalog: HaloRecord, ExtHaloRecord

# Halo catalog: one lossless Float32 dataset per record field + metadata + provenance.
const CATALOG_FORMAT = "huntian-catalog/1"

const _UNITS = Dict(
    "x" => "Mpc/h", "y" => "Mpc/h", "z" => "Mpc/h", "RTHL" => "Mpc/h", "Rf" => "Mpc/h",
    "vx" => "km/s", "vy" => "km/s", "vz" => "km/s", "vx2" => "km/s", "vy2" => "km/s", "vz2" => "km/s")

_record_type(nfields) = nfields == fieldcount(ExtHaloRecord) ? ExtHaloRecord :
                        nfields == fieldcount(HaloRecord) ? HaloRecord : error("no halo record with $nfields fields")

function Huntian.write_halo_catalog(path::AbstractString, halos::Vector{T}; z_out::Real,
                                    provenance::AbstractDict=Dict{String,String}(), finalized::Bool=true,
                                    attributes::AbstractDict=Dict{String,Any}(), compress::Integer=1,
                                    chunk::Integer=1 << 20) where {T<:Union{HaloRecord, ExtHaloRecord}}
    n = length(halos)
    cols = reinterpret(reshape, Float32, halos)             # fieldcount(T) × n view of the records
    h5open(path, "w") do f
        attrs(f)["format"] = CATALOG_FORMAT
        attrs(f)["record"] = T === ExtHaloRecord ? "extended" : "basic"
        attrs(f)["n_halos"] = n
        attrs(f)["z_out"] = Float64(z_out)
        attrs(f)["RTHLmax"] = n > 0 ? Float64(maximum(h.RTHL for h in halos)) : 0.0
        attrs(f)["finalized"] = finalized
        attrs(f)["mass_definition"] = "M = 4π/3 · 2.775e11 Ω_m · RTHL³ [Msun/h], Ω_m total (provenance recipe.config_toml)"
        for (k, v) in attributes
            attrs(f)[string(k)] = v
        end
        for (i, name) in enumerate(fieldnames(T))
            nm = string(name)
            d = create_dataset(f, nm, datatype(Float32), dataspace((n,));
                               chunk=(max(1, min(n, chunk)),), shuffle=true, deflate=compress)
            n > 0 && write(d, cols[i, :])
            u = get(_UNITS, nm, "")
            # before finalize_eulerian, positions are Lagrangian and velocities are displacement fields
            !finalized && startswith(nm, "v") && (u = "Mpc/h (displacement)")
            attrs(d)["units"] = u
        end
        g = create_group(f, "provenance")
        for (k, v) in provenance
            attrs(g)[string(k)] = string(v)
        end
    end
    return path
end

function _check_format(f, path)
    haskey(attrs(f), "format") && attrs(f)["format"] == CATALOG_FORMAT || error("not a $(CATALOG_FORMAT) file: $path")
end

function Huntian.read_halo_catalog(path::AbstractString)
    h5open(path, "r") do f
        _check_format(f, path)
        T = attrs(f)["record"] == "extended" ? ExtHaloRecord : HaloRecord
        n = Int(attrs(f)["n_halos"])
        halos = Vector{T}(undef, n)
        cols = reinterpret(reshape, Float32, halos)
        for (i, name) in enumerate(fieldnames(T))
            n > 0 && (cols[i, :] = read(f[string(name)]))
        end
        meta = Dict{String,Any}(k => attrs(f)[k] for k in keys(attrs(f)))
        prov = Dict{String,String}(k => string(attrs(f["provenance"])[k]) for k in keys(attrs(f["provenance"])))
        return halos, meta, prov
    end
end

function Huntian.read_halo_columns(path::AbstractString, names; range=nothing)
    h5open(path, "r") do f
        _check_format(f, path)
        NamedTuple{Tuple(Symbol.(names))}(Tuple(range === nothing ? read(f[string(nm)]) : f[string(nm)][range]
                                                 for nm in names))
    end
end

# Compact, self-contained catalog: quantised columns + full provenance (see Huntian.write_compact_catalog)
const COMPACT_FORMAT = "huntian-compact-catalog/1"
const COMPACT_FORMATS_READ = (COMPACT_FORMAT, "peakpatch-compact-catalog/1")   # older files carry the peakpatch- name

function _write_q(f, name, vals::AbstractVector{<:Real}, ::Type{T}, step, offset; units="", desc="",
                  compress=4) where {T<:Integer}
    lo, hi = typemin(T), typemax(T)
    q = Vector{T}(undef, length(vals)); nclip = 0
    @inbounds for i in eachindex(vals)
        v = round(Int64, (Float64(vals[i]) - offset) / step)
        (v < lo || v > hi) && (nclip += 1; v = clamp(v, lo, hi))
        q[i] = T(v)
    end
    n = length(q)
    d = create_dataset(f, name, datatype(T), dataspace((n,)); chunk=(max(1, min(n, 1 << 20)),), shuffle=true, deflate=compress)
    write(d, q)
    attrs(d)["step"] = Float64(step); attrs(d)["offset"] = Float64(offset)
    attrs(d)["units"] = units; attrs(d)["description"] = desc; attrs(d)["n_clipped"] = nclip
    nclip > 0 && @warn "write_compact_catalog: $nclip values of $name clipped to the $(T) range"
    return nothing
end

function _read_q(f, name)
    d = f[name]; q = read(d)
    st = Float64(attrs(d)["step"]); of = Float64(attrs(d)["offset"])
    return of .+ st .* Float64.(q)
end

function Huntian.write_compact_catalog(path::AbstractString, halos::Vector{<:Union{HaloRecord, ExtHaloRecord}},
                                          prov::AbstractDict; extra_fields::Bool=false, compress::Integer=4)
    n = length(halos)
    cfg = TOML.parse(get(prov, "recipe.config_toml", ""))
    Om = Float64(get(get(cfg, "cosmology", Dict()), "Om", NaN))
    isfinite(Om) || error("write_compact_catalog: provenance has no [cosmology] Om (needed for the mass definition)")
    rho_m = 2.775e11 * Om                                  # Msun/h per (Mpc/h)^3
    h5open(path, "w") do f
        attrs(f)["format"] = COMPACT_FORMAT
        attrs(f)["n_halos"] = n
        attrs(f)["mass_definition"] = "M = 4π/3 ρ_m R_TH³, ρ_m = 2.775e11 Ω_m Msun/h (Mpc/h)^-3, Ω_m = $(Om); R_TH as in the source catalog (abundance-matched if AM was applied)"
        attrs(f)["coordinates"] = "Eulerian comoving positions [Mpc/h] in the Huntian centred frame; observer in provenance recipe.config_toml [grid] cen*"
        for (c, nm) in ((h -> h.x, "x"), (h -> h.y, "y"), (h -> h.z, "z"))
            v = Float64[c(h) for h in halos]
            off = n > 0 ? floor(minimum(v)) - 1.0 : 0.0
            _write_q(f, nm, v, Int32, 0.01, off; units="Mpc/h", desc="comoving position", compress=compress)
        end
        for (c, nm) in ((h -> h.vx, "vx"), (h -> h.vy, "vy"), (h -> h.vz, "vz"))
            _write_q(f, nm, Float64[c(h) for h in halos], Int16, 1.0, 0.0; units="km/s", desc="peculiar velocity", compress=compress)
        end
        lm = Float64[log10(4π / 3 * rho_m * Float64(h.RTHL)^3) for h in halos]
        _write_q(f, "log10M", lm, UInt16, 1e-4, 10.0; units="log10(Msun/h)", desc="halo mass (see mass_definition)", compress=compress)
        if extra_fields
            eltype(halos) <: ExtHaloRecord || error("extra_fields=true needs an extended (ioutshear ≥ 1) catalog")
            _write_q(f, "zform", Float64[h.zform for h in halos], UInt16, 2e-4, -1.0; desc="formation redshift (collapse of R_TH/2)", compress=compress)
            _write_q(f, "e_v", Float64[h.e_v for h in halos], UInt16, 1e-5, 0.0; desc="ellipticity of the strain at R_TH", compress=compress)
            _write_q(f, "p_v", Float64[h.p_v for h in halos], UInt16, 2e-5, -0.6; desc="prolateness of the strain at R_TH", compress=compress)
        end
        g = create_group(f, "provenance")
        for (k, v) in prov
            attrs(g)[string(k)] = string(v)
        end
    end
    return path
end

function Huntian.read_compact_catalog(path::AbstractString)
    h5open(path, "r") do f
        attrs(f)["format"] in COMPACT_FORMATS_READ || error("not a $(COMPACT_FORMAT) file: $path")
        g = f["provenance"]
        prov = Dict{String,String}(k => string(attrs(g)[k]) for k in keys(attrs(g)))
        cols = Dict{Symbol,Any}()
        for nm in ("x", "y", "z", "vx", "vy", "vz")
            cols[Symbol(nm)] = Float32.(_read_q(f, nm))
        end
        cols[:M] = 10 .^ _read_q(f, "log10M")
        for nm in ("zform", "e_v", "p_v")
            haskey(f, nm) && (cols[Symbol(nm)] = Float32.(_read_q(f, nm)))
        end
        cols[:provenance] = prov
        return (; cols...)
    end
end

end # module HDF5Ext
