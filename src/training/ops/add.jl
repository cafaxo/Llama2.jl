struct AddOp{T,N}
    input_a::Tensor{T,N}
    input_b::Tensor{T,N}
    output::Tensor{T,N}
end

function add(input_a::Tensor{T,N}, input_b::Tensor{T,N}) where {T,N}
    @assert size(input_a.value) == size(input_b.value)

    @assert input_a.graph == input_b.graph
    graph = input_a.graph

    output = Tensor(size(input_a.value), graph)

    push!(graph.operations, AddOp(input_a, input_b, output))
    return output
end

function forward!(op::AddOp)
    @. op.output.value = op.input_a.value + op.input_b.value
    return nothing
end

function backward!(op::AddOp)
    op.input_a.grad .+= op.output.grad
    op.input_b.grad .+= op.output.grad
    return nothing
end

import Base: +

+(a::Tensor, b::Tensor) = add(a, b)
