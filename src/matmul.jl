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

    # FIXME: preallocate this
    x = quantize(block_q8_K, x)

    Threads.@threads for i in 1:length(y)
        y[i] = dot(view(A, :, i), x)
    end

    return nothing
end
