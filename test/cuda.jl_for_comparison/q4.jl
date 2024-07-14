using Llama2: block_q4_K

function dequantize_q4_kernel(y, x, nb, QK_K)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
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
# for the CPU solution.
function get_scale_min_k4(index::Int, scales::Vector{UInt8})
    scale = (scales[index] & 0xF) * 0.1f0  # Replace with actual scale factor
    min = (scales[index] >> 4) * 0.1f0     # Replace with actual min factor
    return scale, min
end

function dequantize_cuda!(y::CuVector{T}, x::CuVector{block_q4_K}) where T <: Union{Float16, Float32}
    k = length(y)
    QK_K = 256  # Assuming QK_K is a known constant
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    threads_per_block = 256
    blocks_per_grid = ceil(Int, nb / threads_per_block)

    @cuda threads=threads_per_block blocks=blocks_per_grid dequantize_q4_kernel(y, x, nb, QK_K)

    return y
end

# NEW get_scale_min_k4 for NTuple{12...
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
