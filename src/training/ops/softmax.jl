struct SoftmaxOp{T}
    input::Tensor{T,2}
    output::Tensor{T,2}
end

function softmax(input::Tensor)
    output = Tensor(size(input.value), input.graph)
    push!(input.graph.operations, SoftmaxOp(input, output))
    return output
end

function forward!(op::SoftmaxOp{T}) where {T}
    y = op.output.value
    x = op.input.value

    for n in axes(x, 2)
        x_max = convert(T, -Inf)

        @turbo for i in axes(x, 1)
            x_max = max(x[i, n], x_max)
        end

        exp_sum = zero(T)

        @turbo for i in axes(x, 1)
            y[i, n] = exp(x[i, n] - x_max)
            exp_sum += y[i, n]
        end

        @turbo for i in axes(x, 1)
            y[i, n] /= exp_sum
        end
    end

    return nothing
end

function backward!(op::SoftmaxOp{T}) where {T}
    ∂x = op.input.grad
    ∂y = op.output.grad
    x = op.input.value

    for n in axes(x, 2)
        x_max = convert(T, -Inf)

        @turbo for i in axes(x, 1)
            x_max = max(x[i, n], x_max)
        end

        exp_sum = zero(T)
        ∂y_dot_exp_x = zero(T)

        @turbo for i in axes(x, 1)
            xi_exp = exp(x[i, n] - x_max)
            exp_sum += xi_exp
            ∂y_dot_exp_x += ∂y[i, n]*xi_exp
        end

        @turbo for i in axes(x, 1)
            xi_exp = exp(x[i, n] - x_max)
            ∂x[i, n] += (∂y[i, n] * exp_sum - ∂y_dot_exp_x) * xi_exp / exp_sum^2
        end
    end

    return nothing
end
