struct Tensor{T,N}
    value::Array{T,N}
    grad::Array{T,N}
    graph::ComputeGraph{T}
end

function Tensor(size::Tuple, graph::ComputeGraph{T}; parameter::Bool = false, name::String = "") where {T}
    if parameter && name == ""
        error("parameters must be named")
    end

    value = zeros(T, size)
    grad = zeros(T, size)

    value_flat = reshape(value, :)
    grad_flat = reshape(grad, :)

    push!(graph.values, value_flat)
    push!(graph.gradients, grad_flat)

    if parameter
        push!(graph.parameter_values, value_flat)
        push!(graph.parameter_gradients, grad_flat)
        graph.parameter_dict[name] = value
    end

    return Tensor(value, grad, graph)
end

function Base.reshape(x::Tensor, size::Tuple)
    return Tensor(reshape(x.value, size), reshape(x.grad, size), x.graph)
end

function Base.reshape(x::Tensor, size...)
    return Tensor(reshape(x.value, size), reshape(x.grad, size), x.graph)
end

@inline Base.size(x::Tensor) = size(x.value)
@inline Base.size(x::Tensor, i::Int) = size(x.value, i)
