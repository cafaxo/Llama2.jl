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

Base.@propagate_inbounds function get_scale_min_k4(j::Int, q::AbstractVector{UInt8})
    if j <= 4
        d = q[j] & UInt8(63)
        m = q[j + 4] & UInt8(63)
    else
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4)
    end

    return d, m
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

function dequantize!(y::AbstractVector{Float32}, x::AbstractVector{block_q4_K})
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
# SOME MORE TESTING for the quantize code...
using CUDA
function dequantize!(y::CuVector{Float32}, x::CuVector{block_q4_K})
    nb = length(y) ÷ QK_K
    threads_per_block = 256  # Or another suitable value based on your GPU
    blocks = ceil(Int, nb / threads_per_block)

    # @show y[1:5]
    CUDA.@cuda blocks=blocks threads=threads_per_block cuda_dequantize!(y, x)
    # @show y[1:5]
end

function cuda_dequantize!(y, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = length(y)
    nb = k ÷ QK_K

    if i <= nb
        d = x[i].d
        dmin = x[i].dmin
        scales = x[i].scales
        q = x[i].qs
        for j in 1:(QK_K ÷ 64)
            sc, m = get_scale_min_k4(2 * (j - 1) + 1, scales)  # This function needs to be adapted for CUDA
            d1 = d * sc
            m1 = dmin * m

            sc, m = get_scale_min_k4(2 * (j - 1) + 2, scales)  # This function needs to be adapted for CUDA
            d2 = d * sc
            m2 = dmin * m

            for l in 1:32
                idx1 = QK_K * (i - 1) + 64 * (j - 1) + l
                idx2 = QK_K * (i - 1) + 64 * (j - 1) + 32 + l
                if idx1 <= k
                    y[idx1] = d1 * ((q[32 * (j - 1) + l]) & 0xF) - m1
                end
                if idx2 <= k
                    y[idx2] = d2 * ((q[32 * (j - 1) + l]) >> 4) - m2
                end
            end
        end
    end
    return
end
# NEW get_scale_min_k4...
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
