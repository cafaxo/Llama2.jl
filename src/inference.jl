@kwdef struct ModelConfig
    dim::Int                # transformer dimension
    hidden_dim::Int         # for ffn layers
    n_layers::Int           # number of layers
    n_heads::Int            # number of query heads
    n_kv_heads::Int         # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::Int         # vocabulary size, usually 256 (byte-level)
    seq_len::Int            # max sequence length
    rope_freq_base::Float32
  end
  
  function Base.show(io::IO, mime::MIME"text/plain", config::ModelConfig)
    println(io, "ModelConfig(")
    println(io, "  dim            = ", config.dim, ",")
    println(io, "  hidden_dim     = ", config.hidden_dim, ",")
    println(io, "  n_layers       = ", config.n_layers, ",")
    println(io, "  n_heads        = ", config.n_heads, ",")
    println(io, "  n_kv_heads     = ", config.n_kv_heads, ",")
    println(io, "  vocab_size     = ", config.vocab_size, ",")
    println(io, "  seq_len        = ", config.seq_len, ",")
    println(io, "  rope_freq_base = ", config.rope_freq_base, ",")
    print(io, ")")
  end
  
  @kwdef struct TransformerLayerWeights
    # weights for rmsnorms
    rms_att_weight::AbstractVector{Float32} # (dim,) # usually Float32
    rms_ffn_weight::AbstractVector{Float32} # (dim,) # usually Float32
    # weights for matmuls
    wq::AbstractMatrix # (dim, dim)
    wk::AbstractMatrix # (dim, dim)
    wv::AbstractMatrix # (dim, dim) # different quantization usually
    wo::AbstractMatrix # (dim, dim)
    # weights for ffn
    w1::AbstractMatrix # (dim, hidden_dim)
    w2::AbstractMatrix # (hidden_dim, dim) # different quantization usually
    w3::AbstractMatrix # (dim, hidden_dim)
  end
  
  @kwdef struct TransformerWeights
    token_embedding_table::AbstractMatrix # (dim, vocab_size)
    layers::Vector{TransformerLayerWeights} # NTuple{TransformerLayerWeights} # but not every layer is the same type!
    # final rmsnorm
    rms_final_weight::AbstractVector # (dim,) # Float32
    output_weight::AbstractMatrix # (dim, vocab_size)
  end
  
  struct LanguageModel{TOK<:Tokenizer}
    config::ModelConfig
    tokenizer::TOK
    weights::TransformerWeights
  end
  
  function Base.show(io::IO, mime::MIME"text/plain", model::LanguageModel)
    println(io, "LanguageModel(")
    show(io, mime, model.config)
    print(io, ")")
  end
  
  struct KVCache
    key_cache::AbstractArray{Float32, 3}   # (head_size, n_heads, seq_len), {Float32,3}
    value_cache::AbstractArray{Float32, 3} # (seq_len, head_size, n_heads), {Float32,3}
  end
  
  KVCache(head_size::Int, n_heads::Int, seq_len::Int, AT) = KVCache(
    AT(zeros(Float32, head_size, n_heads, seq_len)),
    AT(zeros(Float32, seq_len, head_size, n_heads)),
  )
  
  @kwdef struct RunState
    # current wave of activations
    x::AbstractVector{Float32}      # activation at current time stamp (dim,)
    xb::AbstractVector{Float32}     # same, but inside a residual branch (dim,)
    xb2::AbstractVector{Float32}    # an additional buffer just for convenience (dim,)
    hb::AbstractVector{Float32}     # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::AbstractVector{Float32}    # buffer for hidden dimension in the ffn (hidden_dim,)
    q::AbstractVector{Float32}      # query (dim,)
    k::AbstractVector{Float32}      # key (dim,)
    v::AbstractVector{Float32}      # value (dim,)
    att::AbstractVector{Float32}    # buffer for scores/attention values (seq_len * n_heads,)
    logits::AbstractVector{Float32} # output logits
    # kv cache
    kvcache_layers::Vector{KVCache}
  end
  
  RunState(c::ModelConfig, AT) where T = RunState(;
    x              = AT(zeros(Float32, c.dim)),
    xb             = AT(zeros(Float32, c.dim)),
    xb2            = AT(zeros(Float32, c.dim)),
    hb             = AT(zeros(Float32, c.hidden_dim)),
    hb2            = AT(zeros(Float32, c.hidden_dim)),
    q              = AT(zeros(Float32, c.dim)),
    k              = AT(zeros(Float32, (c.dim ÷ c.n_heads) * c.n_kv_heads)),
    v              = AT(zeros(Float32, (c.dim ÷ c.n_heads) * c.n_kv_heads)),
    att            = AT(zeros(Float32, c.seq_len * c.n_heads)),
    logits         = AT(zeros(Float32, c.vocab_size)),
    kvcache_layers = [KVCache(c.dim ÷ c.n_heads, c.n_kv_heads, c.seq_len, AT) for _ in 1:c.n_layers],
  )
  get_run_state(model::LanguageModel) = begin
    ARRAY_TYPE = typeof(model.weights.token_embedding_table)
    RunState(model.config, ARRAY_TYPE.name.wrapper)
end

@kernel function rmsnorm_kernel_optimized!(o, x, weight, length_x)
    local_idx = @index(Local, Linear)
    global_idx = @index(Global, Linear)
    group_size = @groupsize()[1]

    # Shared memory for partial sums and final ss value
    shared_mem = @localmem Float32 (group_size)
    
    ss = 0.0f0
    
    # Only the first workgroup calculates the normalization factor
    # Calculate partial sum of squares
    @inbounds for j in local_idx:group_size:length_x
        ss += x[j] * x[j]
    end
    
    shared_mem[local_idx] = ss
    @synchronize
    
    # Parallel reduction
    s = group_size ÷ 2
    while s > 0
        if local_idx <= s
            shared_mem[local_idx] += shared_mem[local_idx + s]
        end
        @synchronize
        s ÷= 2
    end
    
    # Final calculation
    if local_idx == 1
        ss = shared_mem[1]
        ss /= length_x
        ss += 1f-6
        ss = 1f0 / sqrt(ss)
    end
    # Broadcast ss to all workgroups
    if local_idx == 1
        @inbounds shared_mem[1] = ss
    end

    @synchronize

    # All threads read the broadcasted ss value
    sall = shared_mem[1]
    @synchronize

    # Each thread calculates its corresponding output
    if global_idx <= length_x
        @inbounds o[global_idx] = weight[global_idx] * (sall * x[global_idx])
    end
end

function rmsnorm!(o::AbstractVector, x::AbstractVector, weight::AbstractVector)
    length_x = length(x)
    backend = KernelAbstractions.get_backend(o)
    
    # Choose an appropriate group size (e.g., 256)
    group_size = 256
    
    kernel! = rmsnorm_kernel_optimized!(backend, group_size)
    kernel!(o, x, weight, length_x, ndrange=length_x, )
end

@kernel function rope_kernel_v2!(x, @Const(pos), @Const(head_size_div2), @Const(n_heads), @Const(theta_scale), @Const(freq_scale))
    i, head = @index(Global, NTuple)
    
    if i <= head_size_div2 && head <= n_heads
        theta_base = freq_scale * (pos - 1)
        
        idx = 2 * (i - 1)
        real_part = x[idx + 1, head]
        imag_part = x[idx + 2, head]
        
        theta = theta_base * (theta_scale ^ (i - 1))
        c, s = cos(theta), sin(theta)

        new_real = muladd(real_part, c, -imag_part * s)
        new_imag = muladd(real_part, s, imag_part * c)
        
        x[idx + 1, head] = new_real
        x[idx + 2, head] = new_imag
    end
end

function rope!(x::AbstractMatrix{Float32}, pos::Int, freq_base::Float32)
    head_size, n_heads = size(x)
    head_size_div2 = head_size ÷ 2
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))

    workgroup_size = (16, 16)  # Adjust these values based on your hardware
    kernel! = rope_kernel_v2!(KernelAbstractions.get_backend(x), workgroup_size)
    
    kernel!(x, pos, head_size_div2, n_heads, theta_scale, freq_scale, ndrange=(head_size_div2, n_heads))
end

@kernel function attention_weights_kernel!(att, @Const(key_cache), @Const(q), n_gqa)
    t, h = @index(Global, NTuple)

    if t <= size(att, 1) && h <= size(att, 2)
        key_h = (h - 1) ÷ n_gqa + 1
        s = 0f0

        @inbounds for j in 1:size(q, 1)
            s += q[j, h] * key_cache[j, key_h, t]
        end
        @inbounds att[t, h] = s
    end
end

function attention_weights!(att::AbstractArray, key_cache::AbstractArray, q::AbstractArray)
    n_gqa = size(q, 2) ÷ size(key_cache, 2)

    kernel! = attention_weights_kernel!(KernelAbstractions.get_backend(att))
    kernel!(att, key_cache, q, n_gqa, ndrange=size(att))
end

@kernel function combine_values_kernel!(xb, @Const(value_cache), @Const(att), n_gqa)
    i, h = @index(Global, NTuple)
  
    if i <= size(xb, 1) && h <= size(xb, 2)
        s = 0.0f0
        value_h = 1 + (h - 1) ÷ n_gqa
        
        for t in 1:size(att, 1)
            s += att[t, h] * value_cache[t, i, value_h]
        end
        
        xb[i, h] = s
    end
end

function combine_values!(xb::AbstractMatrix, value_cache::AbstractArray, att::AbstractMatrix)
    n_gqa = size(att, 2) ÷ size(value_cache, 3)
  
    kernel! = combine_values_kernel!(KernelAbstractions.get_backend(xb))
    kernel!(xb, value_cache, att, n_gqa, ndrange=size(xb))
end

@kernel function softmax_kernel_v2!(att, @Const(attention_maximum))
    i, h = @index(Global, NTuple)
    local_idx = @index(Local)
    group_size = @groupsize()[1]

    if h <= size(att, 2)
        max_val = attention_maximum[h]
        exp_sum = 0.0f0
        
        # Shared memory for partial sums
        shared_mem = @localmem Float32 (group_size)

        # Calculate partial exp sum
        for t in local_idx:group_size:size(att, 1)
            exp_val = exp(att[t, h] - max_val)
            exp_sum += exp_val
            att[t, h] = exp_val
        end

        shared_mem[local_idx] = exp_sum
        @synchronize

        # Parallel reduction for exp_sum
        s = group_size ÷ 2
        while s > 0
            if local_idx <= s
                shared_mem[local_idx] += shared_mem[local_idx + s]
            end
            @synchronize
            s ÷= 2
        end

        @synchronize
        exp_sum = shared_mem[1]

        # Normalize
        for t in local_idx:group_size:size(att, 1)
            att[t, h] /= exp_sum
        end
    end
end

@views function softmax_for!(att::AbstractMatrix)
    pos_size, n_heads = size(att) 
    backend = KernelAbstractions.get_backend(att)
    att_max = reshape(maximum(att, dims=1), :)

    group_size = 32  # Adjust based on your hardware
    kernel! = softmax_kernel_v2!(backend, (group_size, 1))
    kernel!(att, att_max, ndrange=(group_size, n_heads), )
end

silu(x) = x*σ(x) # Basically: x * (1f0 / (1f0 + exp(-x)))

@views function transformer!(token::Int, pos::Int, config::ModelConfig, s::RunState, weights::TransformerWeights)
    (;
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        rope_freq_base,
    ) = config
    head_size = dim ÷ n_heads

    # copy the token embedding into x
    x = s.x
    dequantize!(x, weights.token_embedding_table[:, token])

    # forward all the layers
    for l in 1:n_layers
        w = weights.layers[l]
        kv = s.kvcache_layers[l]

        # attention rmsnorm
        rmsnorm!(s.xb, x, w.rms_att_weight)

        # qkv matmuls for this position
        matmul!(s.q, w.wq, s.xb) # [16, 4096, Llama2.block_q4_K]
        matmul!(s.k, w.wk, s.xb) 
        matmul!(s.v, w.wv, s.xb)

        q = reshape(s.q, head_size, n_heads)
        k = reshape(s.k, head_size, n_kv_heads)

        # apply RoPE rotation to the q and k vectors for each head
        rope!(q, pos, rope_freq_base)
        rope!(k, pos, rope_freq_base)

        # save key,value at this time step (pos) to our kv cache
        copyto!(kv.key_cache[:, :, pos], s.k)
        copyto!(kv.value_cache[pos, :, :], s.v)

        # take a contiguous slice of the attention buffer
        att = reshape(s.att[1:(n_heads*pos)], pos, n_heads)

        # multihead attention
        attention_weights!(att, kv.key_cache, q)

        att ./= sqrt(Float32(head_size))

        softmax_for!(att)

        xb = reshape(s.xb, head_size, n_heads)

        # weighted sum of the values
        combine_values!(xb, kv.value_cache, att)

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
        s.hb .= silu.(s.hb)

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
        stop_on_special_token = true,
        max_seq_len = typemax(Int),
        bos_token = true,
    )

    if !bos_token && isempty(prompt)
        error("Prompt cannot be empty if bos_token = false")
    end

    (; config, weights, tokenizer) = model

    prompt_tokens = encode(prompt, tokenizer)

    state = get_run_state(model)

    time_start = time_ns()

    if bos_token
        pushfirst!(prompt_tokens, tokenizer.bos_token_id)
    end

    if !bos_token
        print(tokenizer.id_to_token[prompt_tokens[1]])
    end

    token = prompt_tokens[1]
    generated_seq_len = 0

    for pos in 1:min(config.seq_len, max_seq_len)
        # forward the transformer to get logits for the next token
        transformer!(token, pos, config, state, weights)
        generated_seq_len += 1

        if pos+1 <= length(prompt_tokens)
            next = prompt_tokens[pos+1]
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
                logits = Array(state.logits) # copy to CPU since wsample is not supported on GPU
                next = wsample(1:config.vocab_size, logits)
            end
        end

        if stop_on_special_token && (next == tokenizer.bos_token_id || next == tokenizer.eos_token_id)
            break
        end

        next_str = tokenizer.id_to_token[next]

        #if pos == 1 && length(prompt_tokens) >= 1
        #    # do not print the input padding that we added
        #    next_str = next_str[2:end]
        #end

        print(next_str)

        # advance forward
        token = next
    end

    println()

    # report our achieved tok/s
    time_end = time_ns()
    @printf "-------\nachieved tok/s: %.2f\n" generated_seq_len / (time_end - time_start) * 1e9

    return nothing
end
