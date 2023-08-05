struct AttentionWeightsOp{T}
    Q::Tensor{T,4}
    K::Tensor{T,4}
    att::Tensor{T,4}
end

function attention_weights(Q::Tensor{T,4}, K::Tensor{T,4}) where {T}
    @assert size(Q) == size(K)
    _, n_heads, seq_len, batch_size = size(Q)
    graph = Q.graph

    att = Tensor((seq_len, n_heads, seq_len, batch_size), graph)
    push!(graph.operations, AttentionWeightsOp(Q, K, att))
    return att
end

function forward!(op::AttentionWeightsOp)
    Q = op.Q.value
    K = op.K.value
    att = op.att.value

    @tturbo for n in axes(att, 4), qi in axes(att, 3), ki in axes(att, 1), h in axes(att, 2)
        s = zero(eltype(att))
        for i in axes(Q, 1)
            s += Q[i, h, qi, n] * K[i, h, ki, n]
        end
        att[ki, h, qi, n] = s
    end

    return nothing
end

function backward!(op::AttentionWeightsOp)
    Q = op.Q.value
    ∂Q = op.Q.grad
    K = op.K.value
    ∂K = op.K.grad
    ∂att = op.att.grad

    @tturbo for n in axes(∂att, 4), qi in axes(∂att, 3), ki in axes(∂att, 1), h in axes(∂att, 2), i in axes(∂Q, 1)
        ∂Q[i, h, qi, n] += ∂att[ki, h, qi, n] * K[i, h, ki, n]
    end

    @tturbo for n in axes(∂att, 4), qi in axes(∂att, 3), ki in axes(∂att, 1), h in axes(∂att, 2), i in axes(∂K, 1)
        ∂K[i, h, ki, n] += ∂att[ki, h, qi, n] * Q[i, h, qi, n]
    end

    return nothing
end

struct MaskOp{T}
    att::Tensor{T,4}
end

function mask!(att::Tensor{T,4}) where {T}
    push!(att.graph.operations, MaskOp(att))
    return att
end

function forward!(op::MaskOp)
    att = op.att.value

    for n in axes(att, 4), qi in axes(att, 3), h in axes(att, 2), ki in (qi+1):size(att, 1)
        att[ki, h, qi, n] = -Inf32
    end

    return nothing
end

function backward!(op::MaskOp)
    ∂att = op.att.grad

    for n in axes(∂att, 4), qi in axes(∂att, 3), h in axes(∂att, 2), ki in (qi+1):size(∂att, 1)
        ∂att[ki, h, qi, n] = 0
    end

    return nothing
end

struct CombineValuesOp{T}
    V::Tensor{T,4}
    att::Tensor{T,4}
    output::Tensor{T,4}
end

function combine_values(V::Tensor{T,4}, att::Tensor{T,4}) where {T}
    _, n_heads, seq_len, batch_size = size(V)
    @assert size(att) == (seq_len, n_heads, seq_len, batch_size)
    graph = V.graph

    output = Tensor(size(V), graph)
    push!(graph.operations, CombineValuesOp(V, att, output))
    return output
end

function forward!(op::CombineValuesOp)
    V = op.V.value
    att = op.att.value
    y = op.output.value

    fill!(y, 0)

    @tturbo for n in axes(att, 4), qi in axes(att, 3), ki in axes(att, 1), h in axes(att, 2), i in axes(y, 1)
        y[i, h, qi, n] += att[ki, h, qi, n] * V[i, h, ki, n]
    end

    return nothing
end

function backward!(op::CombineValuesOp)
    V = op.V.value
    ∂V = op.V.grad
    att = op.att.value
    ∂att = op.att.grad
    ∂y = op.output.grad

    @tturbo for n in axes(att, 4), qi in axes(att, 3), ki in axes(att, 1), h in axes(att, 2), i in axes(∂y, 1)
        ∂V[i, h, ki, n] += att[ki, h, qi, n] * ∂y[i, h, qi, n]
    end

    @tturbo for n in axes(att, 4), qi in axes(att, 3), ki in axes(att, 1), h in axes(att, 2)
        s = zero(eltype(∂att))
        for i in axes(∂y, 1)
            s += V[i, h, ki, n] * ∂y[i, h, qi, n]
        end
        ∂att[ki, h, qi, n] += s
    end

    return nothing
end
