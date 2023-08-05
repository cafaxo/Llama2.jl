abstract type AbstractOptimizer end

function step!(optimizer::AbstractOptimizer, graph::ComputeGraph)
    copyto_flat!(optimizer.x, graph.parameter_values)
    copyto_flat!(optimizer.∇x, graph.parameter_gradients)
    _step!(optimizer)
    copyto_unflat!(graph.parameter_values, optimizer.x)

    return nothing
end

struct SGDOptimizer{T} <: AbstractOptimizer
    x::Vector{T}
    ∇x::Vector{T}
    learning_rate::T
end

function SGDOptimizer(parameter_count::Int; learning_rate::Float32 = 0.01f0)
    return SGDOptimizer(
        zeros(Float32, parameter_count),
        zeros(Float32, parameter_count),
        learning_rate,
    )
end

function _step!(optimizer::SGDOptimizer)
    (; x, ∇x, learning_rate) = optimizer

    @. x -= learning_rate * ∇x

    return nothing
end

mutable struct SGDMomentum{T} <: AbstractOptimizer
    const x::Vector{T}
    const ∇x::Vector{T}
    const v::Vector{T}
    t::Int
    const α::T
    const ρ::T
end

function SGDMomentum(
        parameter_count::Int;
        α = 0.1f0,
        ρ = 0.9f0,
    )

    return SGDMomentum(
        zeros(Float32, parameter_count),
        zeros(Float32, parameter_count),
        zeros(Float32, parameter_count),
        0,
        α,
        ρ,
    )
end

function _step!(optimizer::SGDMomentum)
    (; x, ∇x, v, t, α, ρ) = optimizer

    t += 1
    optimizer.t = t

    @. v = ρ*v + ∇x
    @. x -= α * v

    return nothing
end

mutable struct AdamOptimizer{T} <: AbstractOptimizer
    const x::Vector{T}
    const ∇x::Vector{T}
    const m::Vector{T}
    const v::Vector{T}
    t::Int
    const α::T
    const β₁::T
    const β₂::T
    const ε::T
end

function AdamOptimizer(
        parameter_count::Int;
        α  = 3f-4,
        β₁ = 0.9f0,
        β₂ = 0.999f0,
        ε  = 1f-8,
    )

    return AdamOptimizer(
        zeros(Float32, parameter_count),
        zeros(Float32, parameter_count),
        zeros(Float32, parameter_count),
        zeros(Float32, parameter_count),
        0,
        α,
        β₁,
        β₂,
        ε,
    )
end

function _step!(optimizer::AdamOptimizer)
    (; x, ∇x, m, v, t, α, β₁, β₂, ε) = optimizer

    t += 1
    optimizer.t = t

    # warmup
    α *= min(1f0, (1-β₂)*t / 2)

    @. m = β₁ * m + $(1 - β₁) * ∇x
    @. v = β₂ * v + $(1 - β₂) * ∇x^2

    @turbo @. x -= α * (m / $(1 - β₁^t)) / (sqrt(v / $(1 - β₂^t)) + ε)

    return nothing
end
