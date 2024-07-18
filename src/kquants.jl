const QK_K = 256

# 4-bit quantization
# 8 blocks of 32 elements each
# weight is represented as x = a * q + b
# Effectively 4.5 bits per weight
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
struct block_q5_K
    d::Float16                    # super-block scale for quantized scales
    dmin::Float16                 # super-block scale for quantized mins
    scales::NTuple{12,UInt8}      # 8-bit block scales
    qh::NTuple{QK_K÷8,UInt8}      # quants, high 1 bit
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

function dequantize(x::AbstractVector{<:Union{block_q4_K,block_q5_K,block_q6_K}})
    y = zeros(Float32, length(x)*QK_K)
    dequantize!(y, x)
    return y
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

function dequantize!(y::AbstractVector{Float32}, x::AbstractVector{block_q4_K})
    @assert length(x) * QK_K == length(y)

    @inbounds for i in 1:length(x)
        d = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :d)))
        dmin = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :dmin)))
        scales = unsafe_pointer_to_field(x, i, :scales)
        q = unsafe_pointer_to_field(x, i, :qs)

        for j in 1:2
            sc = unsafe_load(scales, 2*(j-1) + 1) & UInt8(63)
            m = unsafe_load(scales, 2*(j-1) + 1 + 4) & UInt8(63)

            d1 = d * sc
            m1 = dmin * m

            sc = unsafe_load(scales, 2*(j-1) + 2) & UInt8(63)
            m = unsafe_load(scales, 2*(j-1) + 2 + 4) & UInt8(63)

            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                y[QK_K*(i-1) + 64*(j-1) + l] = d1 * (unsafe_load(q, 32*(j-1) + l) & 0xF) - m1
            end

            for l in 1:32
                y[QK_K*(i-1) + 64*(j-1) + 32 + l] = d2 * (unsafe_load(q, 32*(j-1) + l) >> 4) - m2
            end
        end

        for j in 3:4
            sc = (unsafe_load(scales, 2*(j-1) + 1+4) & 0xF) | ((unsafe_load(scales, 2*(j-1) + 1-4) >> 6) << 4)
            m = (unsafe_load(scales, 2*(j-1) + 1+4) >>  4) | ((unsafe_load(scales, 2*(j-1) + 1-0) >> 6) << 4)

            d1 = d * sc
            m1 = dmin * m

            sc = (unsafe_load(scales, 2*(j-1) + 2+4) & 0xF) | ((unsafe_load(scales, 2*(j-1) + 2-4) >> 6) << 4)
            m = (unsafe_load(scales, 2*(j-1) + 2+4) >>  4) | ((unsafe_load(scales, 2*(j-1) + 2-0) >> 6) << 4)

            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                y[QK_K*(i-1) + 64*(j-1) + l] = d1 * (unsafe_load(q, 32*(j-1) + l) & 0xF) - m1
            end

            for l in 1:32
                y[QK_K*(i-1) + 64*(j-1) + 32 + l] = d2 * (unsafe_load(q, 32*(j-1) + l) >> 4) - m2
            end
        end
    end

    return y
end

function vecdot(x::AbstractVector{block_q4_K}, y::AbstractVector{Float32})
    @assert length(x) * QK_K == length(y)

    s = zero(Float32)

    @inbounds for i in 1:length(x)
        d = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :d)))
        dmin = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :dmin)))
        scales = unsafe_pointer_to_field(x, i, :scales)
        q = unsafe_pointer_to_field(x, i, :qs)

        for j in 1:2
            sc = unsafe_load(scales, 2*(j-1) + 1) & UInt8(63)
            m = unsafe_load(scales, 2*(j-1) + 1 + 4) & UInt8(63)

            d1 = d * sc
            m1 = dmin * m

            sc = unsafe_load(scales, 2*(j-1) + 2) & UInt8(63)
            m = unsafe_load(scales, 2*(j-1) + 2 + 4) & UInt8(63)

            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 64*(j-1) + l] * (d1 * (unsafe_load(q, 32*(j-1) + l) & 0xF) - m1))
            end

            for l in 1:32
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 64*(j-1) + 32 + l] * (d2 * (unsafe_load(q, 32*(j-1) + l) >> 4) - m2))
            end
        end

        for j in 3:4
            sc = (unsafe_load(scales, 2*(j-1) + 1+4) & 0xF) | ((unsafe_load(scales, 2*(j-1) + 1-4) >> 6) << 4)
            m = (unsafe_load(scales, 2*(j-1) + 1+4) >>  4) | ((unsafe_load(scales, 2*(j-1) + 1-0) >> 6) << 4)

            d1 = d * sc
            m1 = dmin * m

            sc = (unsafe_load(scales, 2*(j-1) + 2+4) & 0xF) | ((unsafe_load(scales, 2*(j-1) + 2-4) >> 6) << 4)
            m = (unsafe_load(scales, 2*(j-1) + 2+4) >>  4) | ((unsafe_load(scales, 2*(j-1) + 2-0) >> 6) << 4)

            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 64*(j-1) + l] * (d1 * (unsafe_load(q, 32*(j-1) + l) & 0xF) - m1))
            end

            for l in 1:32
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 64*(j-1) + 32 + l] * (d2 * (unsafe_load(q, 32*(j-1) + l) >> 4) - m2))
            end
        end
    end

    return s
end

function dequantize!(y::AbstractVector{Float32}, x::AbstractVector{block_q5_K})
    @assert length(x) * QK_K == length(y)

    @inbounds for i in 1:length(x)
        d = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :d)))
        dmin = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :dmin)))
        ql = unsafe_pointer_to_field(x, i, :qs)
        qh = unsafe_pointer_to_field(x, i, :qh)
        scales = unsafe_pointer_to_field(x, i, :scales)

        u1 = UInt8(1)
        u2 = UInt8(2)

        for j in 1:2
            sc = unsafe_load(scales, 2*(j-1) + 1) & UInt8(63)
            m = unsafe_load(scales, 2*(j-1) + 1 + 4) & UInt8(63)

            d1 = d * sc
            m1 = dmin * m

            sc = unsafe_load(scales, 2*(j-1) + 2) & UInt8(63)
            m = unsafe_load(scales, 2*(j-1) + 2 + 4) & UInt8(63)

            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                y[256*(i-1) + 64*(j-1) + l] = d1 * ((unsafe_load(ql, 32*(j-1) + l) & 0xF) + (((unsafe_load(qh, l) & u1) != UInt8(0)) ? UInt8(16) : UInt8(0))) - m1
            end

            for l in 1:32
                y[256*(i-1) + 64*(j-1) + 32 + l] = d2 * ((unsafe_load(ql, 32*(j-1) + l)  >> 4) + (((unsafe_load(qh, l) & u2) != UInt8(0)) ? UInt8(16) : UInt8(0))) - m2
            end

            u1 <<= 2
            u2 <<= 2
        end

        for j in 3:4
            sc = (unsafe_load(scales, 2*(j-1) + 1+4) & 0xF) | ((unsafe_load(scales, 2*(j-1) + 1-4) >> 6) << 4)
            m = (unsafe_load(scales, 2*(j-1) + 1+4) >>  4) | ((unsafe_load(scales, 2*(j-1) + 1-0) >> 6) << 4)

            d1 = d * sc
            m1 = dmin * m

            sc = (unsafe_load(scales, 2*(j-1) + 2+4) & 0xF) | ((unsafe_load(scales, 2*(j-1) + 2-4) >> 6) << 4)
            m = (unsafe_load(scales, 2*(j-1) + 2+4) >>  4) | ((unsafe_load(scales, 2*(j-1) + 2-0) >> 6) << 4)

            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                y[256*(i-1) + 64*(j-1) + l] = d1 * ((unsafe_load(ql, 32*(j-1) + l) & 0xF) + (((unsafe_load(qh, l) & u1) != UInt8(0)) ? UInt8(16) : UInt8(0))) - m1
            end

            for l in 1:32
                y[256*(i-1) + 64*(j-1) + 32 + l] = d2 * ((unsafe_load(ql, 32*(j-1) + l)  >> 4) + (((unsafe_load(qh, l) & u2) != UInt8(0)) ? UInt8(16) : UInt8(0))) - m2
            end

            u1 <<= 2
            u2 <<= 2
        end
    end

    return y
end

function vecdot(x::AbstractVector{block_q5_K}, y::AbstractVector{Float32})
    @assert length(x) * QK_K == length(y)

    s = zero(Float32)

    @inbounds for i in 1:length(x)
        d = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :d)))
        dmin = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :dmin)))
        ql = unsafe_pointer_to_field(x, i, :qs)
        qh = unsafe_pointer_to_field(x, i, :qh)
        scales = unsafe_pointer_to_field(x, i, :scales)

        u1 = UInt8(1)
        u2 = UInt8(2)

        for j in 1:2
            sc = unsafe_load(scales, 2*(j-1) + 1) & UInt8(63)
            m = unsafe_load(scales, 2*(j-1) + 1 + 4) & UInt8(63)

            d1 = d * sc
            m1 = dmin * m

            sc = unsafe_load(scales, 2*(j-1) + 2) & UInt8(63)
            m = unsafe_load(scales, 2*(j-1) + 2 + 4) & UInt8(63)

            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                s = Base.FastMath.add_fast(s, y[256*(i-1) + 64*(j-1) + l] * (d1 * ((unsafe_load(ql, 32*(j-1) + l) & 0xF) + (((unsafe_load(qh, l) & u1) != UInt8(0)) ? UInt8(16) : UInt8(0))) - m1))
            end

            for l in 1:32
                s = Base.FastMath.add_fast(s, y[256*(i-1) + 64*(j-1) + 32 + l] * (d2 * ((unsafe_load(ql, 32*(j-1) + l)  >> 4) + (((unsafe_load(qh, l) & u2) != UInt8(0)) ? UInt8(16) : UInt8(0))) - m2))
            end

            u1 <<= 2
            u2 <<= 2
        end

        for j in 3:4
            sc = (unsafe_load(scales, 2*(j-1) + 1+4) & 0xF) | ((unsafe_load(scales, 2*(j-1) + 1-4) >> 6) << 4)
            m = (unsafe_load(scales, 2*(j-1) + 1+4) >>  4) | ((unsafe_load(scales, 2*(j-1) + 1-0) >> 6) << 4)

            d1 = d * sc
            m1 = dmin * m

            sc = (unsafe_load(scales, 2*(j-1) + 2+4) & 0xF) | ((unsafe_load(scales, 2*(j-1) + 2-4) >> 6) << 4)
            m = (unsafe_load(scales, 2*(j-1) + 2+4) >>  4) | ((unsafe_load(scales, 2*(j-1) + 2-0) >> 6) << 4)

            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                s = Base.FastMath.add_fast(s, y[256*(i-1) + 64*(j-1) + l] * (d1 * ((unsafe_load(ql, 32*(j-1) + l) & 0xF) + (((unsafe_load(qh, l) & u1) != UInt8(0)) ? UInt8(16) : UInt8(0))) - m1))
            end

            for l in 1:32
                s = Base.FastMath.add_fast(s, y[256*(i-1) + 64*(j-1) + 32 + l] * (d2 * ((unsafe_load(ql, 32*(j-1) + l)  >> 4) + (((unsafe_load(qh, l) & u2) != UInt8(0)) ? UInt8(16) : UInt8(0))) - m2))
            end

            u1 <<= 2
            u2 <<= 2
        end
    end

    return s
end

function dequantize!(y::AbstractVector{Float32}, x::AbstractVector{block_q6_K})
    @assert length(x) * QK_K == length(y)

    @inbounds for i in 1:length(x)
        ql = unsafe_pointer_to_field(x, i, :ql)
        qh = unsafe_pointer_to_field(x, i, :qh)
        scales = unsafe_pointer_to_field(x, i, :scales)
        d = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :d)))

        for n in 1:(QK_K÷128)
            for l in 1:16
                q1 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l +  0) & 0xF) | (((unsafe_load(qh, 32*(n-1) + l) >> 0) & 0x3) << 4)) - 32
                q2 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l + 32) & 0xF) | (((unsafe_load(qh, 32*(n-1) + l) >> 2) & 0x3) << 4)) - 32
                q3 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l +  0)  >> 4) | (((unsafe_load(qh, 32*(n-1) + l) >> 4) & 0x3) << 4)) - 32
                q4 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l + 32)  >> 4) | (((unsafe_load(qh, 32*(n-1) + l) >> 6) & 0x3) << 4)) - 32

                is = 8*(n-1) + 0 + 1

                y[QK_K*(i-1) + 128*(n-1) + l +  0] = d * (Int16(unsafe_load(scales, is + 0)) * q1)
                y[QK_K*(i-1) + 128*(n-1) + l + 32] = d * (Int16(unsafe_load(scales, is + 2)) * q2)
                y[QK_K*(i-1) + 128*(n-1) + l + 64] = d * (Int16(unsafe_load(scales, is + 4)) * q3)
                y[QK_K*(i-1) + 128*(n-1) + l + 96] = d * (Int16(unsafe_load(scales, is + 6)) * q4)
            end

            for l in 17:32
                q1 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l +  0) & 0xF) | (((unsafe_load(qh, 32*(n-1) + l) >> 0) & 0x3) << 4)) - 32
                q2 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l + 32) & 0xF) | (((unsafe_load(qh, 32*(n-1) + l) >> 2) & 0x3) << 4)) - 32
                q3 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l +  0)  >> 4) | (((unsafe_load(qh, 32*(n-1) + l) >> 4) & 0x3) << 4)) - 32
                q4 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l + 32)  >> 4) | (((unsafe_load(qh, 32*(n-1) + l) >> 6) & 0x3) << 4)) - 32

                is = 8*(n-1) + 1 + 1

                y[QK_K*(i-1) + 128*(n-1) + l +  0] = d * (Int16(unsafe_load(scales, is + 0)) * q1)
                y[QK_K*(i-1) + 128*(n-1) + l + 32] = d * (Int16(unsafe_load(scales, is + 2)) * q2)
                y[QK_K*(i-1) + 128*(n-1) + l + 64] = d * (Int16(unsafe_load(scales, is + 4)) * q3)
                y[QK_K*(i-1) + 128*(n-1) + l + 96] = d * (Int16(unsafe_load(scales, is + 6)) * q4)
            end
        end
    end

    return y
end

function vecdot(x::AbstractVector{block_q6_K}, y::AbstractVector{Float32})
    @assert length(x) * QK_K == length(y)

    s = zero(Float32)

    @inbounds for i in 1:length(x)
        ql = unsafe_pointer_to_field(x, i, :ql)
        qh = unsafe_pointer_to_field(x, i, :qh)
        scales = unsafe_pointer_to_field(x, i, :scales)
        d = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :d)))

        for n in 1:(QK_K÷128)
            for l in 1:16
                q1 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l +  0) & 0xF) | (((unsafe_load(qh, 32*(n-1) + l) >> 0) & 0x3) << 4)) - 32
                q2 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l + 32) & 0xF) | (((unsafe_load(qh, 32*(n-1) + l) >> 2) & 0x3) << 4)) - 32
                q3 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l +  0)  >> 4) | (((unsafe_load(qh, 32*(n-1) + l) >> 4) & 0x3) << 4)) - 32
                q4 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l + 32)  >> 4) | (((unsafe_load(qh, 32*(n-1) + l) >> 6) & 0x3) << 4)) - 32

                is = 8*(n-1) + 0 + 1

                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 128*(n-1) + l +  0] * (d * (Int16(unsafe_load(scales, is + 0)) * q1)))
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 128*(n-1) + l + 32] * (d * (Int16(unsafe_load(scales, is + 2)) * q2)))
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 128*(n-1) + l + 64] * (d * (Int16(unsafe_load(scales, is + 4)) * q3)))
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 128*(n-1) + l + 96] * (d * (Int16(unsafe_load(scales, is + 6)) * q4)))
            end

            for l in 17:32
                q1 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l +  0) & 0xF) | (((unsafe_load(qh, 32*(n-1) + l) >> 0) & 0x3) << 4)) - 32
                q2 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l + 32) & 0xF) | (((unsafe_load(qh, 32*(n-1) + l) >> 2) & 0x3) << 4)) - 32
                q3 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l +  0)  >> 4) | (((unsafe_load(qh, 32*(n-1) + l) >> 4) & 0x3) << 4)) - 32
                q4 = reinterpret(Int8, (unsafe_load(ql, 64*(n-1) + l + 32)  >> 4) | (((unsafe_load(qh, 32*(n-1) + l) >> 6) & 0x3) << 4)) - 32

                is = 8*(n-1) + 1 + 1

                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 128*(n-1) + l +  0] * (d * (Int16(unsafe_load(scales, is + 0)) * q1)))
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 128*(n-1) + l + 32] * (d * (Int16(unsafe_load(scales, is + 2)) * q2)))
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 128*(n-1) + l + 64] * (d * (Int16(unsafe_load(scales, is + 4)) * q3)))
                s = Base.FastMath.add_fast(s, y[QK_K*(i-1) + 128*(n-1) + l + 96] * (d * (Int16(unsafe_load(scales, is + 6)) * q4)))
            end
        end
    end

    return s
end
