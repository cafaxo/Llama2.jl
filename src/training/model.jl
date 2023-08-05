# create a compute graph so that we can later run forward and backward passes over it
function llama2_graph(x::Tensor; dim::Int, hidden_dim::Int, n_layers::Int, n_heads::Int)
    vocab_size, seq_len, batch_size = size(x)
    @assert dim % n_heads == 0
    head_size = dim ÷ n_heads

    x = reshape(x, vocab_size, :)

    # compute embedded tokens
    x = dense(x; out_features=dim, name="token_embedding_table")

    for l in 1:n_layers
        # attention rmsnorm
        y = rmsnorm(x; name="layer$(l)_rms_att_weight")

        Q = dense(y; out_features=dim, name="layer$(l)_wq")
        K = dense(y; out_features=dim, name="layer$(l)_wk")
        V = dense(y; out_features=dim, name="layer$(l)_wv")

        Q = reshape(Q, head_size, n_heads, seq_len, batch_size)
        K = reshape(K, head_size, n_heads, seq_len, batch_size)
        V = reshape(V, head_size, n_heads, seq_len, batch_size)

        # apply RoPE rotation to the q and k vectors for each head
        rope!(Q)
        rope!(K)

        att = attention_weights(Q, K)
        att = mask!(att)

        # TODO: do inplace
        att = inv(sqrt(Float32(head_size))) * att
        att = reshape(att, seq_len, :)
        att = softmax(att)
        att = reshape(att, seq_len, n_heads, seq_len, batch_size)

        # weighted sum of the values
        y = combine_values(V, att)
        y = reshape(y, dim, :)

        # final matmul to get the output of the attention
        y = dense(y; out_features=dim, name="layer$(l)_wo")

        # residual connection
        x = x + y

        # feed-forward net
        y = rmsnorm(x; name="layer$(l)_rms_ffn_weight")
        hb1 = dense(y; out_features = hidden_dim, name="layer$(l)_w1")
        hb2 = dense(y; out_features = hidden_dim, name="layer$(l)_w3")
        hb = mul(silu(hb1), hb2)
        y = dense(hb; out_features = dim, name="layer$(l)_w2")

        # residual connection
        x = x + y
    end

    # final rmsnorm
    x = rmsnorm(x; name="rms_final_weight")

    # classifier into logits
    logits = dense(x; out_features=vocab_size, name="output_weight")

    return reshape(logits, vocab_size, seq_len, batch_size)
end

function extract_layer_weights(d::Dict, l::Int)
    return TransformerLayerWeights(;
        rms_att_weight = d["layer$(l)_rms_att_weight"],
        rms_ffn_weight = d["layer$(l)_rms_ffn_weight"],
        wq             = d["layer$(l)_wq"],
        wk             = d["layer$(l)_wk"],
        wv             = d["layer$(l)_wv"],
        wo             = d["layer$(l)_wo"],
        w1             = d["layer$(l)_w1"],
        w2             = d["layer$(l)_w2"],
        w3             = d["layer$(l)_w3"],
    )
end

function extract_weights(d::Dict, n_layers::Int)
    return TransformerWeights(;
        token_embedding_table = permutedims(d["token_embedding_table"]),
        layers                = [extract_layer_weights(d, l) for l in 1:n_layers],
        rms_final_weight      = d["rms_final_weight"],
        output_weight         = d["output_weight"],
    )
end

function load_weights!(d::Dict, weights::TransformerWeights)
    for (l, layer) in enumerate(weights.layers)
        copyto!(d["layer$(l)_rms_att_weight"], layer.rms_att_weight)
        copyto!(d["layer$(l)_rms_ffn_weight"], layer.rms_ffn_weight)
        copyto!(d["layer$(l)_wq"], layer.wq)
        copyto!(d["layer$(l)_wk"], layer.wk)
        copyto!(d["layer$(l)_wv"], layer.wv)
        copyto!(d["layer$(l)_wo"], layer.wo)
        copyto!(d["layer$(l)_w1"], layer.w1)
        copyto!(d["layer$(l)_w2"], layer.w2)
        copyto!(d["layer$(l)_w3"], layer.w3)
    end

    copyto!(d["token_embedding_table"], permutedims(weights.token_embedding_table))
    copyto!(d["rms_final_weight"], weights.rms_final_weight)
    copyto!(d["output_weight"], weights.output_weight)

    return nothing
end

function train(
        config::ModelConfig,
        tokens::Vector{Int};
        init_weights::Union{TransformerWeights,Nothing} = nothing,
        n_tokens::Int = 10_000_000,
        batch_size::Int = 16,
    )
    (;
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        seq_len,
    ) = config

    graph = ComputeGraph()
    input = Tensor((vocab_size, seq_len, batch_size), graph)
    output = llama2_graph(input; dim, hidden_dim, n_layers, n_heads)
    target = zeros(Float32, vocab_size, seq_len, batch_size)
    loss = kl_divergence(reshape(output, vocab_size, :), reshape(target, vocab_size, :))

    if !isnothing(init_weights)
        load_weights!(graph.parameter_dict, init_weights)
    end

    optimizer = AdamOptimizer(number_of_parameters(graph); α=1f-3)

    println("Training a model with $(number_of_parameters(graph)) parameters...")

    n_iters = n_tokens ÷ (batch_size*seq_len)

    progress = Progress(n_iters; showspeed=true)
    loss_est = Inf32
    loss_α = 0.99f0

    for iter in 1:n_iters
        fill!(input.value, 0)
        fill!(target, 0)

        for n in 1:batch_size
            offset = rand(0:(length(tokens) - seq_len - 2))

            for i in 1:seq_len
                input.value[tokens[offset+i],i,n] = 1
                target[tokens[offset+i+1],i,n] = 1
            end
        end

        forward!(graph)
        if loss_est < Inf32
            loss_est = (1-loss_α)*loss.value[1] + loss_α*loss_est
        else
            loss_est = loss.value[1]
        end

        zero_gradients!(graph)
        loss.grad[1] = 1
        backward!(graph)

        step!(optimizer, graph)

        ProgressMeter.next!(progress; showvalues = [
            (:iteration, "$iter / $n_iters"),
            (:training_loss, @sprintf("%.3f", loss_est))])
    end

    return extract_weights(graph.parameter_dict, n_layers)
end
