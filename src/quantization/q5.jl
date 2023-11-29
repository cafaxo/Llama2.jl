
function quantize!(y::Vector{block_q5_K}, x::Vector{Float32})
    k = length(x)
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    L = zeros(UInt8, QK_K)
    scales = zeros(Float32, QK_K÷32)
    mins = zeros(Float32, QK_K÷32)
    weights = zeros(Float32, 32)

    for i in 1:nb
        max_scale = 0f0
        max_min = 0f0

        for j in 1:(QK_K÷32)
            # FIXME
            scales[j], mins[j] = make_qkx1_quants(
                15,
                view(x, (QK_K*(i-1) + 32*(j-1) + 1):(QK_K*(i-1) + 32*j)),
                view(L, (32*(j-1) + 1):(32*j)),
                5,
            )

            scale = scales[j]
            if scale > max_scale
                max_scale = scale
            end

            min = mins[j]
            if min > max_min
                max_min = min
            end
        end

        inv_scale = max_scale > 0 ? 63f0/max_scale : 0f0
        inv_min   = max_min   > 0 ? 63f0/max_min   : 0f0

        yi_d = MutableField(Float16, y, i, :d)
        yi_dmin = MutableField(Float16, y, i, :dmin)
        yi_scales = MutableField(UInt8, y, i, :scales)  
        yi_qs = MutableField(UInt8, y, i, :qs)   # qs
        yi_qh = MutableField(UInt8, y, i, :qh)   # qh

        for j in 1:(QK_K÷32)
            ls = Base.unsafe_trunc(UInt8, round(inv_scale*scales[j]))
            lm = Base.unsafe_trunc(UInt8, round(inv_min*mins[j]))
            ls = min(UInt8(63), ls)
            lm = min(UInt8(63), lm)

            if j <= 4
                yi_scales[j] = ls
                yi_scales[j+4] = lm
            else
                yi_scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4)
                yi_scales[j-4] |= ((ls >> 4) << 6)
                yi_scales[j-0] |= ((lm >> 4) << 6)
            end
        end

        yi_d[1] = Float16(max_scale/63f0)
        yi_dmin[1] = Float16(max_min/63f0)

        for j in 1:(QK_K÷32)
            sc, m = get_scale_min_k4(j, yi_scales)

            d = Float32(yi_d[1]) * sc

            if d == 0f0
                continue
            end

            dm = Float32(yi_dmin[1]) * m

            for ii in 1:32
                l = Base.unsafe_trunc(Int32, round((x[QK_K*(i-1) + 32*(j-1) + ii] + dm)/d))
                l = max(Int32(0), min(Int32(15), l))
                L[32*(j-1) + ii] = l
            end
        end

        m1, m2 = 1, 2
        for j in 1:(QK_K ÷ 64)
            for l in 1:32
                l1 = L[64*(j-1) + l]
                if l1 > 15
                    l1 -= 16
                    yi_qh[l] |= m1
                end
                l2 = L[64*(j-1) + l + 32]
                if l2 > 15
                    l2 -= 16
                    yi_qh[l] |= m2
                end
                yi_qs[32*(j-1) + l] = l1 | (l2 << 4)
            end
            # m1 = m1<<2; m2 = m2<<2
        end
    end

    return y
end

function dequantize!(y::AbstractVector{Float32}, x::AbstractVector{block_q5_K})
    k = length(y)
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    @inbounds for i in 1:nb
        d = Float32(MutableField(Float16, x, i, :d)[1])
        dmin = Float32(MutableField(Float16, x, i, :dmin)[1])
        scales = MutableField(UInt8, x, i, :scales)
        q = MutableField(UInt8, x, i, :qs)

        for j in 1:(QK_K÷64)
            sc, m = get_scale_min_k4(2*(j-1) + 1, scales)
            d1 = d * sc
            m1 = dmin * m

            sc, m = get_scale_min_k4(2*(j-1) + 2, scales)
            d2 = d * sc
            m2 = dmin * m

            @simd ivdep for l in 1:32
                y[QK_K*(i-1) + 64*(j-1) + l] = d1 * (q[32*(j-1) + l] & 0xF) - m1
            end

            @simd ivdep for l in 1:32
                y[QK_K*(i-1) + 64*(j-1) + 32 + l] = d2 * (q[32*(j-1) + l] >> 4) - m2
            end
        end
    end

    return y
end
