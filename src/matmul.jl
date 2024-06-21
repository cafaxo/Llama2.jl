# matmul! computes y .= A' * x
function matmul!(
        y::AbstractVector{Float32},
        A::AbstractMatrix{Float32},
        x::AbstractVector{Float32},
    )

    mul!(y, A', x)
    return nothing
end

function matmul!(
    y::Vector{Float32},
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

# Unused, but was used for dequantized weights testing (which turned out to be less accurate numerically)
function matmul_fp16_kernel!(y::CuDeviceVector{Float32}, A::CuDeviceMatrix{Float16}, x::CuDeviceVector{T2}) where T2
    nx, ny = size(A)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= ny
        temp = 0f0
        for i = 1:nx
            temp += A[i, idx] * x[i]
        end
        y[idx] = temp
    end
    return
end
function matmul_fp16!(
    y::CuVector{Float32},
    A::CuMatrix{T},
    x::CuVector{T2},
) where {T<:AbstractFloat, T2<:AbstractFloat}
    nx, ny = size(A)
    block_dim = 256
    grid_dim = ceil(Int, nx / block_dim)

    @cuda blocks=grid_dim threads=block_dim matmul_fp16_kernel(y, A, x)
end


