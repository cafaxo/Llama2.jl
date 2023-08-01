const QK_K = 256

struct block_q4_K
    d::Float16               # super-block scales/mins
    dmin::Float16
    scales::NTuple{12,UInt8} # 4-bit block scales/mins
    qs::NTuple{QK_K÷2,UInt8} # 4-bit quants
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

function dequantize(x::AbstractVector{<:Union{block_q4_K,block_q6_K,block_q8_K}})
    y = zeros(Float32, length(x)*QK_K)
    dequantize!(y, x)
    return y
end

function quantize(::Type{T}, x::AbstractVector{Float32}) where {T<:Union{block_q4_K,block_q6_K,block_q8_K}}
    @assert length(x) % QK_K == 0
    return quantize!(Vector{T}(undef, length(x) ÷ QK_K), x)
end
