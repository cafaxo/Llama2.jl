struct DenseOp{T}
    weights::Tensor{T,2}
    input::Tensor{T,2}
    output::Tensor{T,2}
    name::String
end

function dense(input::Tensor{T,2}; out_features::Int, name::String) where {T}
    graph = input.graph
    in_features = size(input, 1)

    output = Tensor((
        out_features,
        size(input, 2),
    ), graph)

    weights = Tensor((
        in_features,
        out_features,
    ), graph; parameter=true, name)

    k = 1 / in_features
    rand!(Uniform(-sqrt(k), sqrt(k)), weights.value)

    push!(graph.operations, DenseOp(weights, input, output, name))
    return output
end

function forward!(op::DenseOp)
    W = op.weights.value
    x = op.input.value
    y = op.output.value

    @tturbo for i in axes(x, 2), m in axes(W, 2)
        s = zero(eltype(y))
        for k in axes(W, 1)
            s += W[k, m] * x[k, i]
        end
        y[m, i] = s
    end

    return nothing
end

function backward!(op::DenseOp)
    W = op.weights.value
    ∂W = op.weights.grad
    x = op.input.value
    ∂x = op.input.grad
    ∂y = op.output.grad

    # input gradient
    @tturbo for i in axes(∂y, 2), k in axes(W, 1)
        s = zero(eltype(∂x))
        for m in axes(W, 2)
            s += W[k, m] * ∂y[m, i]
        end
        ∂x[k, i] += s
    end

    # weight gradient
    @tturbo for i in axes(∂y, 2), m in axes(∂W, 2), k in axes(∂W, 1)
        ∂W[k, m] += ∂y[m, i] * x[k, i]
    end

    return nothing
end
