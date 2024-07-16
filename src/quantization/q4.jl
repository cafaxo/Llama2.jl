function make_qkx1_quants(nmax::Int, x::AbstractVector{Float32}, L::AbstractVector{UInt8}, ntry::Int)
    min_x = x[1]
    max_x = x[1]

    n = length(x)

    for i in 2:length(x)
        if x[i] < min_x
            min_x = x[i]
        end

        if x[i] > max_x
            max_x = x[i]
        end
    end

    if max_x == min_x
        for i in 1:n
            L[i] = 0
        end

        return 0f0, 0
    end

    if min_x > 0f0
        min_x = 0f0
    end

    iscale = nmax / (max_x - min_x)
    scale = inv(iscale)

    for _ in 1:ntry
        sumlx = 0f0
        suml2 = Int32(0)
        did_change = false

        for i in 1:n
            l = Base.unsafe_trunc(Int32, round(iscale*(x[i] - min_x)))
            l = max(Int32(0), min(Int32(nmax), l))

            if l != L[i]
                L[i] = l
                did_change = true
            end

            sumlx += (x[i] - min_x)*l
            suml2 += l*l
        end

        scale = sumlx/suml2
        sum = 0f0

        for i in 1:n
            sum += x[i] - scale*L[i]
        end

        min_x = sum/n

        if min_x > 0f0
            min_x = 0f0
        end

        iscale = inv(scale)

        if !did_change
            break
        end
    end

    return scale, -min_x
end


function quantize!(y::Vector{block_q4_K}, x::Vector{Float32})
    k = length(x)
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    L = zeros(UInt8, QK_K)
    mins = zeros(Float32, QK_K ÷ 32)
    scales = zeros(Float32, QK_K ÷ 32)

    for i in 1:nb
        max_scale = 0f0
        max_min = 0f0

        for j in 1:(QK_K÷32)
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
        yi_qs = MutableField(UInt8, y, i, :qs)

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

        for j in 1:(QK_K ÷ 64)
            for l in 1:32
                yi_qs[32*(j-1) + l] = L[64*(j-1) + l] | (L[64*(j-1) + 32 + l] << 4)
            end
        end
    end

    return y
end

Base.@propagate_inbounds function get_scale_min_k4(j::Int, q::NTuple{12, UInt8})
    if j <= 4
        d = q[j] & UInt8(63)
        m = q[j + 4] & UInt8(63)
    else
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4)
    end

    return d, m
end

@kernel function dequantize_q4_kernel!(y, x, nb, QK_K)
    idx = @index(Global)
    if idx <= nb
        d = Float32(x[idx].d)
        dmin = Float32(x[idx].dmin)
        scales = x[idx].scales
        q = x[idx].qs

        for j in 1:(QK_K ÷ 64)
            sc1, m1 = get_scale_min_k4(2 * (j - 1) + 1, scales)
            d1 = d * sc1
            min1 = dmin * m1

            sc2, m2 = get_scale_min_k4(2 * (j - 1) + 2, scales)
            d2 = d * sc2
            min2 = dmin * m2

            for l in 1:32
                y[QK_K * (idx - 1) + 64 * (j - 1) + l] = d1 * (q[32 * (j - 1) + l] & 0xF) - min1
                y[QK_K * (idx - 1) + 64 * (j - 1) + 32 + l] = d2 * (q[32 * (j - 1) + l] >> 4) - min2
            end
        end
    end
end

function dequantize!(y::AbstractVector{T}, x::AbstractVector{block_q4_K}) where T <: Union{Float16, Float32}
    k = length(y)
    QK_K = 256  # Assuming QK_K is a known constant
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    kernel_q4 = dequantize_q4_kernel!(KernelAbstractions.get_backend(y), 64)
    kernel_q4(y, x, nb, QK_K, ndrange=nb)

    return y
end
