using Llama2: block_q6_K

# ChatGPT-4o generated CUDA kernel based on the aboce dequantize! function
function dequantize_q6_kernel(y, x, nb, QK_K)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= nb
        ql = x[idx].ql
        qh = x[idx].qh
        scales = x[idx].scales
        d = Float32(x[idx].d)

        for n in 1:(QK_K ÷ 128)
            for l in 1:32
                q1 = reinterpret(Int8, (ql[64 * (n - 1) + l + 0] & 0xF) | (((qh[32 * (n - 1) + l] >> 0) & 0x3) << 4)) - 32
                q2 = reinterpret(Int8, (ql[64 * (n - 1) + l + 32] & 0xF) | (((qh[32 * (n - 1) + l] >> 2) & 0x3) << 4)) - 32
                q3 = reinterpret(Int8, (ql[64 * (n - 1) + l + 0] >> 4) | (((qh[32 * (n - 1) + l] >> 4) & 0x3) << 4)) - 32
                q4 = reinterpret(Int8, (ql[64 * (n - 1) + l + 32] >> 4) | (((qh[32 * (n - 1) + l] >> 6) & 0x3) << 4)) - 32

                is = 8 * (n - 1) + (l - 1) ÷ 16 + 1

                y[QK_K * (idx - 1) + 128 * (n - 1) + l + 0] = d * (Int16(scales[is + 0]) * q1)
                y[QK_K * (idx - 1) + 128 * (n - 1) + l + 32] = d * (Int16(scales[is + 2]) * q2)
                y[QK_K * (idx - 1) + 128 * (n - 1) + l + 64] = d * (Int16(scales[is + 4]) * q3)
                y[QK_K * (idx - 1) + 128 * (n - 1) + l + 96] = d * (Int16(scales[is + 6]) * q4)
            end
        end
    end
end

function dequantize_cuda!(y::CuVector{T}, x::CuVector{block_q6_K}) where T <: Union{Float16, Float32}
    k = length(y)
    QK_K = 256  # Assuming QK_K is a known constant
    @assert k % QK_K == 0
    nb = k ÷ QK_K

    threads_per_block = 256
    blocks_per_grid = ceil(Int, nb / threads_per_block)

    @cuda threads=threads_per_block blocks=blocks_per_grid dequantize_q6_kernel(y, x, nb, QK_K)

    return y
end
