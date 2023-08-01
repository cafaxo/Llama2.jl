@kwdef struct ModelConfig
    dim::Int        # transformer dimension
    hidden_dim::Int # for ffn layers
    n_layers::Int   # number of layers
    n_heads::Int    # number of query heads
    n_kv_heads::Int # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::Int # vocabulary size, usually 256 (byte-level)
    seq_len::Int    # max sequence length
end

function Base.show(io::IO, mime::MIME"text/plain", config::ModelConfig)
    println(io, "ModelConfig(")
    println(io, "  dim         = ", config.dim, ",")
    println(io, "  hidden_dim  = ", config.hidden_dim, ",")
    println(io, "  n_layers    = ", config.n_layers, ",")
    println(io, "  n_heads     = ", config.n_heads, ",")
    println(io, "  n_kv_heads  = ", config.n_kv_heads, ",")
    println(io, "  vocab_size  = ", config.vocab_size, ",")
    println(io, "  seq_len     = ", config.seq_len, ",")
    print(io, ")")
end

@kwdef struct TransformerLayerWeights{Q}
    # weights for rmsnorms
    rms_att_weight::Vector{Float32} # (dim,)
    rms_ffn_weight::Vector{Float32} # (dim,)
    # weights for matmuls
    wq::Matrix{Q} # (dim, dim)
    wk::Matrix{Q} # (dim, dim)
    wv::Matrix{Q} # (dim, dim)
    wo::Matrix{Q} # (dim, dim)
    # weights for ffn
    w1::Matrix{Q} # (dim, hidden_dim)
    w2::Matrix{Q} # (hidden_dim, dim)
    w3::Matrix{Q} # (dim, hidden_dim)
end

@kwdef struct TransformerWeights{Q,OW}
    token_embedding_table::Matrix{Q} # (dim, vocab_size)
    layers::Vector{TransformerLayerWeights{Q}}
    # final rmsnorm
    rms_final_weight::Vector{Float32} # (dim,)
    output_weight::Matrix{OW} # (dim, vocab_size)
end

struct LanguageModel{TW<:TransformerWeights}
    config::ModelConfig
    tokenizer::Tokenizer
    weights::TW
end

function Base.show(io::IO, mime::MIME"text/plain", model::LanguageModel)
    println(io, "LanguageModel(")
    show(io, mime, model.config)
    print(io, ")")
end

struct KVCache
    key_cache::Array{Float32,3}   # (head_size, n_heads, seq_len)
    value_cache::Array{Float32,3} # (head_size, n_heads, seq_len)
end

KVCache(head_size::Int, n_heads::Int, seq_len::Int) = KVCache(
    zeros(Float32, head_size, n_heads, seq_len),
    zeros(Float32, head_size, n_heads, seq_len),
)

@kwdef struct RunState
    # current wave of activations
    x::Vector{Float32}      # activation at current time stamp (dim,)
    xb::Vector{Float32}     # same, but inside a residual branch (dim,)
    xb2::Vector{Float32}    # an additional buffer just for convenience (dim,)
    hb::Vector{Float32}     # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Vector{Float32}    # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Vector{Float32}      # query (dim,)
    k::Vector{Float32}      # key (dim,)
    v::Vector{Float32}      # value (dim,)
    att::Vector{Float32}    # buffer for scores/attention values (seq_len,)
    logits::Vector{Float32} # output logits
    # kv cache
    kvcache_layers::Vector{KVCache}
end

RunState(c::ModelConfig) = RunState(;
    x              = zeros(Float32, c.dim),
    xb             = zeros(Float32, c.dim),
    xb2            = zeros(Float32, c.dim),
    hb             = zeros(Float32, c.hidden_dim),
    hb2            = zeros(Float32, c.hidden_dim),
    q              = zeros(Float32, c.dim),
    k              = zeros(Float32, c.dim),
    v              = zeros(Float32, c.dim),
    att            = zeros(Float32, c.seq_len),
    logits         = zeros(Float32, c.vocab_size),
    kvcache_layers = [KVCache(c.dim ÷ c.n_heads, c.n_heads, c.seq_len) for _ in 1:c.n_layers],
)

function rmsnorm!(o, x, weight)
    ss = dot(x, x)
    ss /= length(x)
    ss += 1f-5
    ss = 1f0 / sqrt(ss)
    # normalize and scale
    o .= weight .* (ss .* x)
    return nothing
end

function rope!(x::AbstractMatrix{Float32}, pos::Int)
    x = reinterpret(ComplexF32, x)
    head_size_div2, n_heads = size(x)

    freq_base = 10000.0f0
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))

    @inbounds for head in 1:n_heads
        theta = freq_scale * (pos - 1)

        for i in 1:head_size_div2
            x[i, head] *= cis(theta)
            theta *= theta_scale
        end
    end

    return nothing
end

function softmax!(x)
    x .= exp.(x .- maximum(x))
    # normalize
    x ./= sum(x)
    return nothing
end

@views function transformer!(token::Int, pos::Int, config::ModelConfig, s::RunState, weights::TransformerWeights)
    x = s.x

    (;
        dim,
        hidden_dim,
        n_layers,
        n_heads,
    ) = config

    head_size = dim ÷ n_heads

    # copy the token embedding into x
    dequantize!(x, weights.token_embedding_table[:, token])

    # forward all the layers
    for l in 1:n_layers
        w = weights.layers[l]
        kv = s.kvcache_layers[l]

        # attention rmsnorm
        rmsnorm!(s.xb, x, w.rms_att_weight)

        # qkv matmuls for this position
        matmul!(s.q, w.wq, s.xb)
        matmul!(s.k, w.wk, s.xb)
        matmul!(s.v, w.wv, s.xb)

        q = reshape(s.q, head_size, n_heads)
        k = reshape(s.k, head_size, n_heads)

        # apply RoPE rotation to the q and k vectors for each head
        rope!(q, pos)
        rope!(k, pos)

        # save key,value at this time step (pos) to our kv cache
        copyto!(kv.key_cache[:, :, pos], s.k)
        copyto!(kv.value_cache[:, :, pos], s.v)

        # multihead attention. iterate over all heads
        for h in 1:n_heads
            # get the query vector for this head
            q_h = q[:, h]
            # iterate over all timesteps, including the current one
            for t in 1:pos
                # get the key vector for this head and at this timestep
                k_h = kv.key_cache[:, h, t]
                # calculate the attention score as the dot product of q and k
                score = dot(q_h, k_h) / sqrt(Float32(head_size))
                # save the score to the attention buffer
                s.att[t] = score
            end

            # softmax the scores to get attention weights, from 0..pos inclusively
            softmax!(s.att[1:pos])

            xb = reshape(s.xb, head_size, n_heads)

            # weighted sum of the values, store back into xb
            mul!(xb[:, h], kv.value_cache[:, h, 1:pos], s.att[1:pos])
        end

        # final matmul to get the output of the attention
        matmul!(s.xb2, w.wo, s.xb)

        # residual connection back into x
        x .+= s.xb2

        # ffn rmsnorm
        rmsnorm!(s.xb, x, w.rms_ffn_weight)

        # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        # first calculate self.w1(x) and self.w3(x)
        matmul!(s.hb, w.w1, s.xb)
        matmul!(s.hb2, w.w3, s.xb)

        # F.silu silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 1:hidden_dim
            s.hb[i] = s.hb[i] * (1f0 / (1f0 + exp(-s.hb[i])))
        end

        s.hb .*= s.hb2

        # final matmul to get the output of the ffn
        matmul!(s.xb, w.w2, s.hb)

        # residual connection
        x .+= s.xb
    end

    # final rmsnorm
    rmsnorm!(x, x, weights.rms_final_weight)

    # classifier into logits
    matmul!(s.logits, weights.output_weight, x)

    return nothing
end

function sample(
        model::LanguageModel,
        prompt::String = "";
        temperature::Float32 = 0.9f0,
    )

    (; config, weights, tokenizer) = model

    prompt_tokens = bpe_encode(prompt, tokenizer)

    state = RunState(config)

    time_start = time_ns()

    token = 2 # BOS token in Llama-2 sentencepiece

    println("<s>")

    for pos in 1:config.seq_len
        # forward the transformer to get logits for the next token
        transformer!(token, pos, config, state, weights)

        if pos <= length(prompt_tokens)
            next = prompt_tokens[pos]
        else
            # sample the next token
            if temperature == 0f0
                # greedy argmax sampling
                next = argmax(state.logits)
            else
                # apply the temperature to the logits
                state.logits ./= temperature
                # apply softmax to the logits to get the probabilities for next token
                softmax!(state.logits)
                # sample from this distribution to get the next token
                next = wsample(1:config.vocab_size, state.logits)
            end
        end

        print(tokenizer.vocab[next])

        # advance forward
        token = next
    end

    println()

    # report our achieved tok/s
    time_end = time_ns()
    @printf "achieved tok/s: %f\n" config.seq_len / (time_end - time_start)*1e9

    return nothing
end
