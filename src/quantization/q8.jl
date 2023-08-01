function quantize!(y::AbstractVector{block_q8_K}, x::AbstractVector{Float32})
    k = length(x)
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    @inbounds for i in 1:nb
        max = 0f0
        amax = 0f0

        @simd ivdep for j in 1:QK_K
            ax = abs(x[(i-1)*QK_K + j])

            if ax > amax
                amax = ax
                max = x[(i-1)*QK_K + j]
            end
        end

        d = MutableField(Float32, y, i, :d)
        qs = MutableField(Int8, y, i, :qs)
        bsums = MutableField(Int16, y, i, :bsums)

        if amax == 0f0
            d[1] = 0f0

            @simd ivdep for j in 1:QK_K
                qs[j] = Int8(0)
            end

            continue
        end

        iscale = -128f0 / max

        @simd ivdep for j in 1:QK_K
            v = unsafe_trunc(Int32, round(iscale*x[(i-1)*QK_K + j]))
            qs[j] = unsafe_trunc(Int8, min(Int32(127), v))
        end

        for j in 1:(QK_K÷16)
            s = Int16(0)

            for ii in 1:16
                s += qs[16*(j-1) + ii]
            end

            bsums[j] = s
        end

        d[1] = inv(iscale)
    end

    return y
end

function dequantize!(y::Vector{Float32}, x::Vector{block_q8_K})
    k = length(y)
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    @inbounds for i in 1:nb
        d = MutableField(Float32, x, i, :d)
        qs = MutableField(Int8, x, i, :qs)

        @simd ivdep for j in 1:QK_K
            y[QK_K*(i-1) + j] = d[1] * qs[j]
        end
    end

    return y
end
