struct MulOp{T,N}
    input_a::Tensor{T,N}
    input_b::Tensor{T,N}
    output::Tensor{T,N}
end

function mul(input_a::Tensor{T,N}, input_b::Tensor{T,N}) where {T,N}
    @assert size(input_a.value) == size(input_b.value)

    @assert input_a.graph == input_b.graph
    graph = input_a.graph

    output = Tensor(size(input_a.value), graph)

    push!(graph.operations, MulOp(input_a, input_b, output))
    return output
end

function forward!(op::MulOp)
    @. op.output.value = op.input_a.value * op.input_b.value
    return nothing
end

function backward!(op::MulOp)
    input_a = op.input_a.value
    ∂input_a = op.input_a.grad

    input_b = op.input_b.value
    ∂input_b = op.input_b.grad

    ∂output  = op.output.grad

    @. ∂input_a += ∂output * input_b
    @. ∂input_b += ∂output * input_a

    return nothing
end
