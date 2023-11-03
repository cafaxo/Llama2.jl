const QK_K = 256

struct block_q4_K
    d::Float16               # super-block scales/mins
    dmin::Float16
    scales::NTuple{12,UInt8} # 4-bit block scales/mins
    qs::NTuple{QK_K÷2,UInt8} # 4-bit quants
end

# 5-bit quantization
# 8 blocks of 32 elements each
# weight is represented as x = a * q + b
# Effectively 5.5 bits per weight
# struct block_q5_K
#     d::Float16                    # super-block scales
#     scales::NTuple{QK_K÷16,Int8}  # 8-bit block scales
#     qh::NTuple{QK_K÷8,UInt8}      # quants, high bit
#     qs::NTuple{QK_K÷2,UInt8}      # quants, low 4 bits
# end

struct block_q5_K
    d::Float16                    # super-block scale for quantized scales
    dmin::Float16                 # super-block scale for quantized mins
    scales::NTuple{12,UInt8}      # 8-bit block scales
    qh::NTuple{QK_K÷8,UInt8}      # quants, high bit
    qs::NTuple{QK_K÷2,UInt8}      # quants, low 4 bits
end

# 6-bit quantization
# weight is represented as x = a * q
# 16 blocks of 16 elements each
# Effectively 6.5625 bits per weight
struct block_q6_K
    ql::NTuple{QK_K÷2,UInt8}      # quants, lower 4 bits
    qh::NTuple{QK_K÷4,UInt8}      # quants, upper 2 bits
    scales::NTuple{QK_K÷16,Int8}  # scales, quantized with 8 bits
    d::Float16                    # super-block scale
end

# This is only used for intermediate quantization and dot products
struct block_q8_K
    d::Float32                   # delta
    qs::NTuple{QK_K,Int8}        # quants
    bsums::NTuple{QK_K÷16,Int16} # sum of quants in groups of 16
end

function dequantize!(y::AbstractVector{Float32}, x::AbstractVector{Float32})
    copyto!(y, x)
    return y
end

function dequantize(x::AbstractVector{<:Union{block_q4_K,block_q5_K,block_q6_K,block_q8_K}})
    y = zeros(Float32, length(x)*QK_K)
    dequantize!(y, x)
    return y
end

function quantize(::Type{T}, x::AbstractVector{Float32}) where {T<:Union{block_q4_K,block_q5_K,block_q6_K,block_q8_K}}
    @assert length(x) % QK_K == 0
    return quantize!(Vector{T}(undef, length(x) ÷ QK_K), x)
end

@inline _memcpy!(dst, src, n) = ccall(:memcpy, Cvoid, (Ptr{UInt8}, Ptr{UInt8}, Csize_t), dst, src, n)

@inline function reinterpret_nonprimitive(::Type{Out}, x) where {Out}
    In = typeof(x)
    if !isbitstype(Out)
        error("reinterpret target type must be isbits")
    end
    if !isbitstype(In)
        error("reinterpret source type must be isbits")
    end

    in = Ref{In}(x)
    ptr_in = Base.unsafe_convert(Ptr{In}, in)
    out = Ref{Out}()
    ptr_out = Base.unsafe_convert(Ptr{Out}, out)
    GC.@preserve in out begin
        _memcpy!(ptr_out, ptr_in, sizeof(Out))
    end
    return out[]
end


# https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.h
