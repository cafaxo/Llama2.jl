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
    y::AbstractVector{Float32},
    A::AbstractMatrix{T},
    x::AbstractVector{Float32},
) where {T<:Union{block_q4_K,block_q5_K,block_q6_K}}
    if T <: Union{block_q4_K,block_q5_K}
        x = to_block_f16_sums32(x) # FIXME: preallocate this
    else # block_q6_K
        x = to_block_f16_sums16(x) # FIXME: preallocate this
    end

    Threads.@threads for i in 1:length(y)
        y[i] = vecdot(view(A, :, i), x)
    end

    return nothing
end
