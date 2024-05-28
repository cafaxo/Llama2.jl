# matmul! computes y .= A' * x
using Boilerplate
function matmul!(
        y::AbstractVector{Float32},
        A::AbstractMatrix{Float32},
        x::AbstractVector{Float32},
    )

    mul!(y, A', x)
    return nothing
end

function matmul!(
    y::AbstractVector{Float32},
    A::AbstractMatrix{T},
    x::AbstractVector{Float32},
) where {T<:Union{block_q4_K,block_q5_K,block_q6_K}}
    if T <: Union{block_q4_K,block_q5_K}
        x = to_block_f16_sums32(x) # FIXME: preallocate this
    else # block_q6_K
        x = to_block_f16_sums16(x) # FIXME: preallocate this
    end
    # x = quantize(block_q8_K, x)

    Threads.@threads for i in 1:length(y)
        y[i] = vecdot(view(A, :, i), x)
    end

    return nothing
end
# SOME TESTING... WORK IN PROGRESS...
using CUDA

function cuda_matmul2!(y, A, x)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # @cushow row
    if row <= size(A, 1)
        tmp_sum = 0.0f0
        # @cushow tmp_sum
        for col in 1:size(A, 2)
            # @cushow A[row, col]
            # @cushow x[col]
            # tmp_sum += A[row, col] * x[col]
            # @cushow tmp_sum
        end
        y[row] = tmp_sum
    end
    return
end
function matmul!(
    y::CuVector{Float32},
    A::CuArray{T, 2},
    x::CuVector{Float32},
) where {T<:Union{block_q4_K,block_q5_K,block_q6_K}}

    threads_per_block = 256  # Adjust based on your GPU's capabilities
    blocks = ceil(Int, size(A, 1) / threads_per_block)

    CUDA.@cuda blocks=blocks threads=threads_per_block cuda_matmul2!(y, A, x)
    # # FIXME: preallocate this
    # @show block_q8_K
    # @typeof x
    # @typeof A
    # x = quantize(block_q8_K, x)
    # @typeof x

    # for i in 1:length(y)
    #     y[i] = dot(view(A, :, i), x)
    # end

    return nothing
end


