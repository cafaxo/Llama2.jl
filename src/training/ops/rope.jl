struct RopeOp{T}
    input::Tensor{T,4}
end

function rope!(input::Tensor{T,4}) where {T}
    push!(input.graph.operations, RopeOp(input))
    return input
end

function forward!(op::RopeOp)
    x = op.input.value
    x = reinterpret(ComplexF32, x)
    head_size_div2 = size(x, 1)

    freq_base = 10000.0f0
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))

    # loopvec does not support complex numbers yet
    @inbounds @fastmath for n in axes(x, 4), pos in axes(x, 3), h in axes(x, 2)
        theta = freq_scale * (pos - 1)

        for i in axes(x, 1)
            x[i, h, pos, n] *= cis(theta)
            theta *= theta_scale
        end
    end

    return nothing
end

function backward!(op::RopeOp)
    ∂x = op.input.grad
    ∂x = reinterpret(ComplexF32, ∂x)
    head_size_div2 = size(∂x, 1)

    freq_base = 10000.0f0
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))

    # loopvec does not support complex numbers yet
    @inbounds @fastmath for n in axes(∂x, 4), pos in axes(∂x, 3), h in axes(∂x, 2)
        theta = freq_scale * (pos - 1)

        for i in axes(∂x, 1)
            ∂x[i, h, pos, n] *= cis(theta)'
            theta *= theta_scale
        end
    end

    return nothing
end
