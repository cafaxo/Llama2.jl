function make_qx_quants_wrong(nmax::Int, x::AbstractVector{Float32})
    max_ = 0f0
    amax = 0f0

    for i in 1:length(x)
        ax = abs(x[i])
        if ax > amax
            amax = ax
            max_ = x[i]
        end
    end

    if amax == 0f0
        return 0f0
    end

    iscale = -nmax / max_
    return inv(iscale)
end

function quantize!(y::Vector{block_q6_K}, x::Vector{Float32})
    k = length(x)
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    L = zeros(UInt8, QK_K)
    scales = zeros(Float32, QK_K÷16)

    for i in 1:nb
        max_scale = 0f0
        max_abs_scale = 0f0

        for ib in 1:(QK_K÷16)
            # FIXME
            scale = make_qx_quants_wrong(
                32,
                view(x, (QK_K*(i-1) + 16*(ib-1) + 1):(QK_K*(i-1) + 16*ib)),
            )

            scales[ib] = scale

            abs_scale = abs(scale)

            if abs_scale > max_abs_scale
                max_abs_scale = abs_scale
                max_scale = scale
            end
        end

        ql = MutableField(UInt8, y, i, :ql)
        qh = MutableField(UInt8, y, i, :qh)
        yi_scales = MutableField(Int8, y, i, :scales)
        yi_d = MutableField(Float16, y, i, :d)

        iscale = -128f0/max_scale
        yi_d[1] = Float16(inv(iscale))

        for ib in 1:(QK_K÷16)
            yi_scales[ib] = Int8(min(127, unsafe_trunc(Int32, round(iscale*scales[ib]))))
        end

        for j in 1:(QK_K÷16)
            d = Float32(yi_d[1]) * yi_scales[j]

            if d == 0f0
                continue
            end

            for ii in 1:16
                l = unsafe_trunc(Int32, round(x[QK_K*(i-1) + 16*(j-1) + ii] / d))
                l = max(-32, min(31, l))
                L[16*(j-1) + ii] = l + 32
            end
        end

        for j in 1:(QK_K÷128)
            for l in 1:32
                q1 = L[128*(j-1) + l +  0] & 0xF
                q2 = L[128*(j-1) + l + 32] & 0xF
                q3 = L[128*(j-1) + l + 64] & 0xF
                q4 = L[128*(j-1) + l + 96] & 0xF

                ql[64*(j-1) + l + 0] = q1 | (q3 << 4)
                ql[64*(j-1) + l + 32] = q2 | (q4 << 4)
                qh[32*(j-1) + l] = (L[128*(j-1) + l] >> 4) | ((L[128*(j-1) + l + 32] >> 4) << 2) | ((L[128*(j-1) + l + 64] >> 4) << 4) | ((L[128*(j-1) + l + 96] >> 4) << 6)
            end
        end
    end

    return y
end

@kernel function dequantize_q6_kernel!(y, @Const(x), @Const(nb), @Const(QK_K))
    idx = @index(Global)
    if idx <= nb
        ql = x[idx].ql
        qh = x[idx].qh
        scales = x[idx].scales
        d = Float32(x[idx].d)

        for n in 1:2 # QK_K ÷ 128 = 2 
            for l in 1:32
                q1 = reinterpret(Int8, (ql[64*(n-1) + l +  0] & 0xF) | (((qh[32*(n-1) + l] >> 0) & 0x3) << 4)) - 32
                q2 = reinterpret(Int8, (ql[64*(n-1) + l + 32] & 0xF) | (((qh[32*(n-1) + l] >> 2) & 0x3) << 4)) - 32
                q3 = reinterpret(Int8, (ql[64*(n-1) + l +  0]  >> 4) | (((qh[32*(n-1) + l] >> 4) & 0x3) << 4)) - 32
                q4 = reinterpret(Int8, (ql[64*(n-1) + l + 32]  >> 4) | (((qh[32*(n-1) + l] >> 6) & 0x3) << 4)) - 32

                is = 8*(n-1) + (l-1) ÷ 16 + 1

                y[QK_K*(i-1) + 128*(n-1) + l +  0] = d * (Int16(scales[is + 0]) * q1)
                y[QK_K*(i-1) + 128*(n-1) + l + 32] = d * (Int16(scales[is + 2]) * q2)
                y[QK_K*(i-1) + 128*(n-1) + l + 64] = d * (Int16(scales[is + 4]) * q3)
                y[QK_K*(i-1) + 128*(n-1) + l + 96] = d * (Int16(scales[is + 6]) * q4)
            end
        end
    end
end

function dequantize!(y::AbstractVector{T}, x::AbstractVector{block_q6_K}) where T <: Union{Float16, Float32}
    k = length(y)
    QK_K = 256  # Assuming QK_K is a known constant
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    kernel! = dequantize_q6_kernel!(KernelAbstractions.get_backend(y))
    kernel!(y, x, nb, QK_K, ndrange=nb)

    return y
end
