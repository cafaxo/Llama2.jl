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
        A::AbstractMatrix{block_q4_K},
        x::AbstractVector{Float32},
    )

    # FIXME: preallocate this
    x = quantize(block_q8_K, x)

    Threads.@threads for i in 1:length(y)
        y[i] = dot(view(A, :, i), x)
    end

    return nothing
end

# FIXME: needs to be optimized with custom dot method
function matmul!(
        y::AbstractVector{Float32},
        A::AbstractMatrix{block_q6_K},
        x::AbstractVector{Float32},
    )

    tmp = zeros(Float32, length(x))

    for i in 1:length(y)
        dequantize!(tmp, view(A, :, i))
        y[i] = dot(tmp, x)
    end

    return nothing
end
