struct RmsNormOp{T}
    input::Tensor{T,2}
    output::Tensor{T,2}
    weights::Tensor{T,1}
    tmp::Vector{T}
    tmp2::Vector{T}
end

function rmsnorm(input::Tensor{T,2}; name::String) where {T}
    graph = input.graph
    output = Tensor(size(input), graph)

    weights = Tensor((size(input, 1),), graph; parameter=true, name)
    fill!(weights.value, 1)

    push!(graph.operations, RmsNormOp(
        input,
        output,
        weights,
        zeros(T, size(input, 2)),
        zeros(T, size(input, 2)),
    ))
    return output
end

function forward!(op::RmsNormOp)
    x = op.input.value
    y = op.output.value
    w = op.weights.value
    tmp = op.tmp

    @turbo for n in axes(x, 2)
        s = zero(eltype(x))
        for i in axes(x, 1)
            s += x[i, n]^2
        end
        tmp[n] = inv(sqrt(s / size(x, 1) + 1f-5))
    end

    @turbo for n in axes(x, 2), i in axes(x, 1)
        y[i, n] = (tmp[n] * w[i]) * x[i, n]
    end

    return nothing
end

function backward!(op::RmsNormOp)
    x = op.input.value
    ∂x = op.input.grad
    ∂y = op.output.grad
    w = op.weights.value
    ∂w = op.weights.grad
    tmp = op.tmp
    tmp2 = op.tmp2

    @turbo for n in axes(x, 2)
        s = zero(eltype(x))
        for i in axes(x, 1)
            s += w[i] * x[i, n] * ∂y[i, n]
        end
        tmp2[n] = s
    end

    sc = inv(size(x, 1))

    @turbo for n in axes(x, 2), i in axes(x, 1)
        ∂x[i, n] += w[i] * ∂y[i, n] * tmp[n] - (sc * tmp2[n] * tmp[n]^3) * x[i, n]
        ∂w[i] += tmp[n] * x[i, n] * ∂y[i, n]
    end

    return nothing
end
