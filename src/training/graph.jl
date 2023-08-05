struct ComputeGraph{T}
    operations::Vector{Any}
    values::Vector{Vector{T}}
    gradients::Vector{Vector{T}}
    parameter_values::Vector{Vector{T}}
    parameter_gradients::Vector{Vector{T}}
    parameter_dict::Dict{String,Any}
end

ComputeGraph() = ComputeGraph(
    Any[],
    Vector{Float32}[],
    Vector{Float32}[],
    Vector{Float32}[],
    Vector{Float32}[],
    Dict{String,Any}(),
)

number_of_parameters(graph::ComputeGraph) = sum(length, graph.parameter_values; init=0)

function forward!(graph::ComputeGraph)
    for op in graph.operations
        forward!(op)::Nothing
    end

    return nothing
end

function backward!(graph::ComputeGraph)
    for op in Iterators.reverse(graph.operations)
        backward!(op)::Nothing
    end

    return nothing
end

function zero_gradients!(graph::ComputeGraph)
    for grad in graph.gradients
        fill!(grad, 0)
    end

    return nothing
end

function copyto_flat!(x::Vector, vectors::Vector{<:Vector})
    @assert length(x) == sum(length, vectors; init=0)

    n = 0

    for vector in vectors
        copyto!(view(x, n+1:n+length(vector)), vector)
        n += length(vector)
    end

    return nothing
end

function copyto_unflat!(vectors::Vector{<:Vector}, x::Vector)
    @assert length(x) == sum(length, vectors; init=0)

    n = 0

    for vector in vectors
        copyto!(vector, view(x, n+1:n+length(vector)))
        n += length(vector)
    end

    return nothing
end
