struct PointwiseOp{F,DF,T,N}
    f::F
    ∂f::DF
    input::Tensor{T,N}
    output::Tensor{T,N}
end

function pointwise(f, ∂f, input::Tensor)
    output = Tensor(size(input.value), input.graph)
    push!(input.graph.operations, PointwiseOp(f, ∂f, input, output))
    return output
end

function forward!(op::PointwiseOp)
    y = op.output.value
    x = op.input.value

    @turbo for i in eachindex(y, x)
        y[i] = op.f(x[i])
    end

    return nothing
end

function backward!(op::PointwiseOp)
    ∂x = op.input.grad
    ∂y = op.output.grad
    x = op.input.value

    @turbo for i in eachindex(∂x)
        ∂x[i] += op.∂f(x[i]) * ∂y[i]
    end

    return nothing
end

@inline function relu(x::Real)
    y = zero(x)
    return ifelse(x < y, y, x)
end

@inline function ∂relu(x::Real)
    y = zero(x)
    return ifelse(x < y, y, one(x))
end

relu(input::Tensor) = pointwise(relu, ∂relu, input)

#@inline sigmoid(x::Real) = sigmoid_fast(x)
@inline sigmoid(x::Real) = 1 / (1 + exp(-x))

@inline function ∂sigmoid(x::Real)
    σ = sigmoid(x)
    return σ*(1-σ)
end

sigmoid(input::Tensor) = pointwise(sigmoid, ∂sigmoid, input)

function silu(x::Real)
    return x * sigmoid(x)
end

function ∂silu(x::Real)
    σ = sigmoid(x)
    return σ*(1+x*(1-σ))
end

silu(input::Tensor) = pointwise(silu, ∂silu, input)

import Base: *

*(λ::Number, x::Tensor) = pointwise(Base.Fix2(*, λ), Returns(λ), x)
