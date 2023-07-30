function rmsnorm!(o, x, weight)
    ss = dot(x, x)
    ss /= length(x)
    ss += 1.0f-5
    ss = 1.0f0 / sqrt(ss)
    # normalize and scale
    o .= weight .* (ss .* x)
    return nothing
end

function softmax!(x)
    x .= exp.(x .- maximum(x))
    # normalize
    x ./= sum(x)
    return nothing
end

@views function transformer!(token::Int,
    pos::Int,
    p::Config,
    s::RunState,
    w::TransformerWeights)
    # a few convenience variables
    x = s.x
    dim = p.dim
    hidden_dim = p.hidden_dim
    head_size = dim ÷ p.n_heads

    # copy the token embedding into x
    copyto!(x, w.token_embedding_table[:, token])

    # pluck out the "pos" row of freq_cis_real and freq_cis_imag
    freq_cis_real_row = w.freq_cis_real[:, pos]
    freq_cis_imag_row = w.freq_cis_imag[:, pos]

    # forward all the layers
    for l in 1:(p.n_layers)
        # attention rmsnorm
        rmsnorm!(s.xb, x, w.rms_att_weight[:, l])

        # qkv matmuls for this position
        mul!(s.q, w.wq[:, :, l]', s.xb)
        mul!(s.k, w.wk[:, :, l]', s.xb)
        mul!(s.v, w.wv[:, :, l]', s.xb)

        # apply RoPE rotation to the q and k vectors for each head
        for h in 1:(p.n_heads)
            # get the q and k vectors for this head
            q = s.q[((h - 1) * head_size + 1):(h * head_size)]
            k = s.k[((h - 1) * head_size + 1):(h * head_size)]
            # rotate q and k by the freq_cis_real and freq_cis_imag
            for i in 1:(head_size ÷ 2)
                q0 = q[2 * i - 1]
                q1 = q[2 * i]
                k0 = k[2 * i - 1]
                k1 = k[2 * i]
                fcr = freq_cis_real_row[i]
                fci = freq_cis_imag_row[i]
                q[2 * i - 1] = q0 * fcr - q1 * fci
                q[2 * i] = q0 * fci + q1 * fcr
                k[2 * i - 1] = k0 * fcr - k1 * fci
                k[2 * i] = k0 * fci + k1 * fcr
            end
        end

        # save key,value at this time step (pos) to our kv cache
        copyto!(s.key_cache[:, pos, l], s.k)
        copyto!(s.value_cache[:, pos, l], s.v)

        # multihead attention. iterate over all heads
        for h in 1:(p.n_heads)
            # get the query vector for this head
            q = s.q[((h - 1) * head_size + 1):(h * head_size)]
            # iterate over all timesteps, including the current one
            for t in 1:pos
                # get the key vector for this head and at this timestep
                k = s.key_cache[((h - 1) * head_size + 1):(h * head_size), t, l]
                # calculate the attention score as the dot product of q and k
                score = dot(q, k) / sqrt(Float32(head_size))
                # save the score to the attention buffer
                s.att[t] = score
            end

            # softmax the scores to get attention weights, from 0..pos inclusively
            softmax!(s.att[1:pos])

            # weighted sum of the values, store back into xb
            mul!(s.xb[((h - 1) * head_size + 1):(h * head_size)],
                s.value_cache[((h - 1) * head_size + 1):(h * head_size), 1:pos, l],
                s.att[1:pos])
        end

        # final matmul to get the output of the attention
        mul!(s.xb2, w.wo[:, :, l]', s.xb)

        # residual connection back into x
        x .+= s.xb2

        # ffn rmsnorm
        rmsnorm!(s.xb, x, w.rms_ffn_weight[:, l])

        # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        # first calculate self.w1(x) and self.w3(x)
        mul!(s.hb, w.w1[:, :, l]', s.xb)
        mul!(s.hb2, w.w3[:, :, l]', s.xb)

        # F.silu silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 1:hidden_dim
            s.hb[i] = s.hb[i] * (1.0f0 / (1.0f0 + exp(-s.hb[i])))
        end

        s.hb .*= s.hb2

        # final matmul to get the output of the ffn
        mul!(s.xb, w.w2[:, :, l]', s.hb)

        # residual connection
        x .+= s.xb
    end

    # final rmsnorm
    rmsnorm!(x, x, w.rms_final_weight)

    # classifier into logits
    mul!(s.logits, w.token_embedding_table', x)

    return nothing
end

function sample(checkpoint_filename::AbstractString,
    tokenizer_filename::AbstractString;
    temperature::Float32 = 0.9f0)
    config = nothing
    weights = nothing

    # read in the files
    model = load_model(checkpoint_filename, tokenizer_filename)

    # unpack
    (; config, weights) = model
    (; vocab) = model.tokenizer

    # start the inference
    state = RunState(config)

    time_start = time_ns()

    token = 2 # idx=2 is BOS token in Llama-2 sentencepiece

    for pos in 1:(config.seq_len)
        # forward the transformer to get logits for the next token
        transformer!(token, pos, config, state, weights)

        # sample the next token
        if temperature == 0.0f0
            # greedy argmax sampling
            next = argmax(state.logits)
        else
            # apply the temperature to the logits
            state.logits ./= temperature
            # apply softmax to the logits to get the probabilities for next token
            softmax!(state.logits)
            # sample from this distribution to get the next token
            next = wsample(1:(config.vocab_size), state.logits)
        end

        print(String(copy(vocab[next])))

        # advance forward
        token = next
    end
    print('\n')

    # report our achieved tok/s
    time_end = time_ns()
    @printf "achieved tok/s: %f\n" config.seq_len / (time_end - time_start)*1e9

    return nothing
end

"""
    sample(model::TrainedModel, prompt::AbstractString;
    temperature::Float32 = 0.9f0)

Generates a text from the `model`, starting with the given `prompt`.
Returns `nothing`

# Arguments
- `model::TrainedModel`: the loaded model to sample from
- `prompt::AbstractString`: the prompt to start with
- `temperature::Float32`: the temperature to use for sampling (default 0.9); `0.0` will be greedy

# Example
```julia
julia> using Llama

julia> model = load_model("stories42M.bin", "tokenizer.bin")

julia> sample(model, "Julia is the best"; temperature = 0.5f0)
```
"""
function sample(model::TrainedModel, prompt::AbstractString = "";
    temperature::Float32 = 0.9f0)

    # unpack
    (; config, weights) = model
    (; vocab, vocab_scores) = model.tokenizer

    # checks
    @assert length(prompt)<config.seq_len "Maximum output sequence length already by the prompt!"
    @assert temperature>=0 "Temperature must be non-negative!"

    # encode the prompt into tokens
    if !isempty(prompt)
        prompt_tokens = bpe_encode(prompt, vocab, vocab_scores, config.vocab_size)
        num_prompt_tokens = length(prompt_tokens)
    else
        prompt_tokens, num_prompt_tokens = [], 0
    end

    # start the inference
    state = RunState(config)

    time_start = time_ns()

    token = 2 # idx=2 is BOS token in Llama-2 sentencepiece
    println("<s>") # for symmetry (stylistic)

    for pos in 1:(config.seq_len)
        # forward the transformer to get logits for the next token
        transformer!(token, pos, config, state, weights)

        if pos <= num_prompt_tokens
            # if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos]
        else
            # sample the next token
            if temperature == 0.0f0
                # greedy argmax sampling
                next = argmax(state.logits)
            else
                # apply the temperature to the logits
                state.logits ./= temperature
                # apply softmax to the logits to get the probabilities for next token
                softmax!(state.logits)
                # sample from this distribution to get the next token
                next = wsample(1:(config.vocab_size), state.logits)
            end
        end

        # TODO: following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89 in llama2.c)
        # C version: char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next]+1 : vocab[next];
        print(String(copy(vocab[next])))

        # advance forward
        token = next
    end
    print('\n')

    # report our achieved tok/s
    time_end = time_ns()
    @printf "achieved tok/s: %f\n" config.seq_len / (time_end - time_start)*1e9

    return nothing
end
