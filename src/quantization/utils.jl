@inline function vwidemul(x::Vec{N,T}, y::Vec{N,T}) where {N,T}
    WT = widen(T)
    Vec{N,WT}(x) * Vec{N,WT}(y)
end

@inline function vpaddq(x::Vec{8,T}, y::Vec{8,T}) where {T}
    shufflevector(x, y, Val((0, 2, 4, 6, 8, 10, 12, 14))) +
    shufflevector(x, y, Val((1, 3, 5, 7, 9, 11, 13, 15)))
end

@noinline Base.@assume_effects :total function fieldoffset_sym(::Type{T}, s::Symbol) where {T}
    for i in 1:fieldcount(T)
        if fieldname(T, i) == s
            return fieldoffset(T, i)
        end
    end

    return nothing
end

@inline function unsafe_pointer_to_field(y::AbstractVector, index::Int, field::Symbol)
    T = eltype(fieldtype(eltype(y), field))
    return convert(Ptr{T}, pointer(y, index) + fieldoffset_sym(eltype(y), field))
end

# this is a hack that lets us efficiently load and mutate fields of
# an immutable struct that is stored inside a Vector

struct MutableField{T,P<:AbstractVector} <: AbstractVector{T}
    y::P
    index::Int
    fieldoffset::UInt64
    len::Int
end

Base.length(mf::MutableField) = mf.len
Base.eltype(mf::MutableField{T,P}) where {T,P} = T

@inline function MutableField(::Type{T}, y::P, index::Int, field::Symbol) where {T,P<:AbstractVector}
    return MutableField{T,P}(
        y,
        index,
        fieldoffset_sym(eltype(y), field),
        sizeof(fieldtype(eltype(y), field)) ÷ sizeof(T),
    )
end

function Base.getindex(mf::MutableField{T,P}, i::Int) where {T,P}
    y = mf.y

    GC.@preserve y begin
        p = convert(Ptr{T}, pointer(y, mf.index) + mf.fieldoffset) + (i - 1)*sizeof(T)
        unsafe_load(p)
    end
end

function Base.setindex!(mf::MutableField{T,P}, x::T, i::Int) where {T,P}
    y = mf.y

    GC.@preserve y begin
        p = convert(Ptr{T}, pointer(y, mf.index) + mf.fieldoffset) + (i - 1)*sizeof(T)
        unsafe_store!(p, x)
    end
end
