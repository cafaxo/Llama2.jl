

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
