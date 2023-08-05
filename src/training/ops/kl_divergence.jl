struct KLDivergenceOp{T}
    prediction::Tensor{T,2}
    target::Array{T,2}
    output::Tensor{T,1} # FIXME: this should be 0-dimensional
end

function kl_divergence(prediction::Tensor{T,2}, target::Array{T,2}) where {T}
    @assert size(prediction) == size(target)

    graph = prediction.graph
    output = Tensor((1,), graph)

    push!(graph.operations, KLDivergenceOp(prediction, target, output))
    return output
end

function forward!(op::KLDivergenceOp)
    prediction = op.prediction.value
    target = op.target

    s = 0f0

    for n in axes(target, 2)
        exp_max = maximum(view(prediction, :, n))
        # FIXME: no need to allocate here
        log_exp_sum = log(sum(exp, view(prediction, :, n) .- exp_max))

        for i in axes(target, 1)
            p = prediction[i, n] - exp_max
            t = target[i, n]

            s += t > 0 ? t*(log(t) - (p - log_exp_sum)) : 0f0
        end
    end

    op.output.value[1] = s / size(target, 2)
    return nothing
end

function backward!(op::KLDivergenceOp)
    ∂prediction = op.prediction.grad
    prediction = op.prediction.value
    target = op.target

    α = op.output.grad[1] / size(target, 2)

    for n in axes(∂prediction, 2)
        exp_max = maximum(view(prediction, :, n))
        # FIXME: no need to allocate here
        exp_sum = sum(exp, view(prediction, :, n) .- exp_max)

        for i in axes(target, 1)
            p = prediction[i, n] - exp_max
            t = target[i, n]

            ∂prediction[i, n] += α*(exp(p) / exp_sum - t)
        end
    end

    return nothing
end
