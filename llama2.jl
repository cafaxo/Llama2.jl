# this file contains a Julia port of https://github.com/karpathy/llama2.c
# all credit goes to Andrej Karpathy (author of llama2.c)

using LinearAlgebra
using StatsBase
using Printf

# Transformer and RunState structs, and related memory management

struct Config
    dim::Int        # transformer dimension
    hidden_dim::Int # for ffn layers
    n_layers::Int   # number of layers
    n_heads::Int    # number of query heads
    n_kv_heads::Int # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::Int # vocabulary size, usually 256 (byte-level)
    seq_len::Int    # max sequence length
end

read_config(f::IOStream) = Config(
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
)

@kwdef struct TransformerWeights
    # token embedding table
    token_embedding_table::Matrix{Float32} # (vocab_size, dim)
    # weights for rmsnorms
    rms_att_weight::Matrix{Float32} # (layer, dim) rmsnorm weights
    rms_ffn_weight::Matrix{Float32} # (layer, dim)
    # weights for matmuls
    wq::Array{Float32,3} # (layer, dim, dim)
    wk::Array{Float32,3} # (layer, dim, dim)
    wv::Array{Float32,3} # (layer, dim, dim)
    wo::Array{Float32,3} # (layer, dim, dim)
    # weights for ffn
    w1::Array{Float32,3} # (layer, hidden_dim, dim)
    w2::Array{Float32,3} # (layer, dim, hidden_dim)
    w3::Array{Float32,3} # (layer, hidden_dim, dim)
    # final rmsnorm
    rms_final_weight::Vector{Float32} # (dim,)
    # freq_cis for RoPE relatively positional embeddings
    freq_cis_real::Matrix{Float32} # (seq_len, dim/2)
    freq_cis_imag::Matrix{Float32} # (seq_len, dim/2)
end

TransformerWeights(p::Config) = TransformerWeights(;
    token_embedding_table = zeros(Float32, p.dim, p.vocab_size),
    rms_att_weight        = zeros(Float32, p.dim, p.n_layers),
    rms_ffn_weight        = zeros(Float32, p.dim, p.n_layers),
    wq                    = zeros(Float32, p.dim, p.dim, p.n_layers),
    wk                    = zeros(Float32, p.dim, p.dim, p.n_layers),
    wv                    = zeros(Float32, p.dim, p.dim, p.n_layers),
    wo                    = zeros(Float32, p.dim, p.dim, p.n_layers),
    w1                    = zeros(Float32, p.dim, p.hidden_dim, p.n_layers),
    w2                    = zeros(Float32, p.hidden_dim, p.dim, p.n_layers),
    w3                    = zeros(Float32, p.dim, p.hidden_dim, p.n_layers),
    rms_final_weight      = zeros(Float32, p.dim),
    freq_cis_real         = zeros(Float32, (p.dim ÷ p.n_heads) ÷ 2, p.seq_len),
    freq_cis_imag         = zeros(Float32, (p.dim ÷ p.n_heads) ÷ 2, p.seq_len),
)

function checkpoint_init_weights!(w::TransformerWeights, f::IOStream)
    read!(f, w.token_embedding_table)
    read!(f, w.rms_att_weight)
    read!(f, w.wq)
    read!(f, w.wk)
    read!(f, w.wv)
    read!(f, w.wo)
    read!(f, w.rms_ffn_weight)
    read!(f, w.w1)
    read!(f, w.w2)
    read!(f, w.w3)
    read!(f, w.rms_final_weight)
    read!(f, w.freq_cis_real)
    read!(f, w.freq_cis_imag)
    return nothing
end

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
    key_cache::Array{Float32,3}   # (layer, seq_len, dim)
    value_cache::Array{Float32,3} # (layer, seq_len, dim)
end

RunState(p::Config) = RunState(;
    x           = zeros(Float32, p.dim),
    xb          = zeros(Float32, p.dim),
    xb2         = zeros(Float32, p.dim),
    hb          = zeros(Float32, p.hidden_dim),
    hb2         = zeros(Float32, p.hidden_dim),
    q           = zeros(Float32, p.dim),
    k           = zeros(Float32, p.dim),
    v           = zeros(Float32, p.dim),
    att         = zeros(Float32, p.seq_len),
    logits      = zeros(Float32, p.vocab_size),
    key_cache   = zeros(Float32, p.dim, p.seq_len, p.n_layers),
    value_cache = zeros(Float32, p.dim, p.seq_len, p.n_layers),
)

function rmsnorm!(o, x, weight)
    # calculate sum of squares
    ss = dot(x, x)
    ss /= length(x)
    ss += 1f-5
    ss = 1f0 / sqrt(ss)
    # normalize and scale
    o .= weight .* (ss .* x)
    return nothing
end

function softmax!(x)
    # find max value (for numerical stability)
    max_val = maximum(x)
    # exp
    x .= exp.(x .- max_val)
    # normalize
    x ./= sum(x)
    return nothing
end

@views function transformer!(token::Int, pos::Int, p::Config, s::RunState, w::TransformerWeights)
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
    for l in 1:p.n_layers
        # attention rmsnorm
        rmsnorm!(s.xb, x, w.rms_att_weight[:, l])

        # qkv matmuls for this position
        mul!(s.q, w.wq[:, :, l]', s.xb)
        mul!(s.k, w.wk[:, :, l]', s.xb)
        mul!(s.v, w.wv[:, :, l]', s.xb)

        # apply RoPE rotation to the q and k vectors for each head
        for h in 1:p.n_heads
            # get the q and k vectors for this head
            q = s.q[((h-1) * head_size + 1):(h * head_size)]
            k = s.k[((h-1) * head_size + 1):(h * head_size)]
            # rotate q and k by the freq_cis_real and freq_cis_imag
            for i in 1:(head_size ÷ 2)
                q0 = q[2*i-1]
                q1 = q[2*i]
                k0 = k[2*i-1]
                k1 = k[2*i]
                fcr = freq_cis_real_row[i]
                fci = freq_cis_imag_row[i]
                q[2*i-1] = q0 * fcr - q1 * fci
                q[2*i]   = q0 * fci + q1 * fcr
                k[2*i-1] = k0 * fcr - k1 * fci
                k[2*i]   = k0 * fci + k1 * fcr
            end
        end

        # save key,value at this time step (pos) to our kv cache
        copyto!(s.key_cache[:, pos, l], s.k)
        copyto!(s.value_cache[:, pos, l], s.v)

        # multihead attention. iterate over all heads
        for h in 1:p.n_heads
            # get the query vector for this head
            q = s.q[((h-1) * head_size + 1):(h * head_size)]
            # iterate over all timesteps, including the current one
            for t in 1:pos
                # get the key vector for this head and at this timestep
                k = s.key_cache[((h-1) * head_size + 1):(h * head_size), t, l]
                # calculate the attention score as the dot product of q and k
                score = dot(q, k) / sqrt(Float32(head_size))
                # save the score to the attention buffer
                s.att[t] = score
            end

            # softmax the scores to get attention weights, from 0..pos inclusively
            softmax!(s.att[1:pos])

            # weighted sum of the values, store back into xb
            mul!(
                s.xb[((h-1) * head_size + 1):(h * head_size)],
                s.value_cache[((h-1) * head_size + 1):(h * head_size), 1:pos, l],
                s.att[1:pos],
            )
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
            s.hb[i] = s.hb[i] * (1f0 / (1f0 + exp(-s.hb[i])))
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

function main(
        checkpoint_filename::AbstractString,
        tokenizer_filename::AbstractString;
        temperature::Float32 = 0.9f0,
    )

    config = nothing
    weights = nothing

    # read in the model.bin file
    open(checkpoint_filename) do file
        config = read_config(file)
        weights = TransformerWeights(config)
        checkpoint_init_weights!(weights, file)
    end

    # read in the tokenizer.bin file
    vocab = Vector{Vector{UInt8}}(undef, config.vocab_size)
    vocab_scores = Vector{Float32}(undef, config.vocab_size)
    max_token_length = 1

    open(tokenizer_filename) do file
        max_token_length = read(file, Int32)
        for i in 1:config.vocab_size
            vocab_scores[i] = read(file, Float32)
            len = read(file, Int32)
            vocab[i] = read(file, len)
        end
    end

    # create and init the application RunState
    state = RunState(config)

    # the current position we are in
    time_start = time_ns()

    token = 1 # 1 = BOS token in Llama-2 sentencepiece

    for pos in 1:config.seq_len
        # forward the transformer to get logits for the next token
        transformer!(token, pos, config, state, weights)

        # sample the next token
        if temperature == 0f0
            # greedy argmax sampling
            next = argmax(state.logits)
        else
            # apply the temperature to the logits
            state.logits ./= temperature
            # apply softmax to the logits to get the probabilities for next token
            softmax!(state.logits)
            # we now want to sample from this distribution to get the next token
            next = wsample(1:config.vocab_size, state.logits)
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
