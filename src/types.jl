"Configuration for the transformer model"
struct Config
    dim::Int        # transformer dimension
    hidden_dim::Int # for ffn layers
    n_layers::Int   # number of layers
    n_heads::Int    # number of query heads
    n_kv_heads::Int # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::Int # vocabulary size, usually 256 (byte-level)
    seq_len::Int    # max sequence length
end

@kwdef struct TransformerWeights
    token_embedding_table::Matrix{Float32} # (dim, vocab_size)
    # weights for rmsnorms
    rms_att_weight::Matrix{Float32} # (dim, layer)
    rms_ffn_weight::Matrix{Float32} # (dim, layer)
    # weights for matmuls
    wq::Array{Float32, 3} # (dim, dim, layer)
    wk::Array{Float32, 3} # (dim, dim, layer)
    wv::Array{Float32, 3} # (dim, dim, layer)
    wo::Array{Float32, 3} # (dim, dim, layer)
    # weights for ffn
    w1::Array{Float32, 3} # (dim, hidden_dim, layer)
    w2::Array{Float32, 3} # (hidden_dim, dim, layer)
    w3::Array{Float32, 3} # (dim, hidden_dim, layer)
    # final rmsnorm
    rms_final_weight::Vector{Float32} # (dim,)
    # freq_cis for RoPE relative positional embeddings
    freq_cis_real::Matrix{Float32} # ((dim / n_heads) / 2, seq_len)
    freq_cis_imag::Matrix{Float32} # ((dim / n_heads) / 2, seq_len)
end

function TransformerWeights(p::Config)
    TransformerWeights(;
        token_embedding_table = zeros(Float32, p.dim, p.vocab_size),
        rms_att_weight = zeros(Float32, p.dim, p.n_layers),
        rms_ffn_weight = zeros(Float32, p.dim, p.n_layers),
        wq = zeros(Float32, p.dim, p.dim, p.n_layers),
        wk = zeros(Float32, p.dim, p.dim, p.n_layers),
        wv = zeros(Float32, p.dim, p.dim, p.n_layers),
        wo = zeros(Float32, p.dim, p.dim, p.n_layers),
        w1 = zeros(Float32, p.dim, p.hidden_dim, p.n_layers),
        w2 = zeros(Float32, p.hidden_dim, p.dim, p.n_layers),
        w3 = zeros(Float32, p.dim, p.hidden_dim, p.n_layers),
        rms_final_weight = zeros(Float32, p.dim),
        freq_cis_real = zeros(Float32, (p.dim ÷ p.n_heads) ÷ 2, p.seq_len),
        freq_cis_imag = zeros(Float32, (p.dim ÷ p.n_heads) ÷ 2, p.seq_len))
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
    key_cache::Array{Float32, 3}   # (dim, seq_len, layer)
    value_cache::Array{Float32, 3} # (dim, seq_len, layer)
end

function RunState(p::Config)
    RunState(;
        x = zeros(Float32, p.dim),
        xb = zeros(Float32, p.dim),
        xb2 = zeros(Float32, p.dim),
        hb = zeros(Float32, p.hidden_dim),
        hb2 = zeros(Float32, p.hidden_dim),
        q = zeros(Float32, p.dim),
        k = zeros(Float32, p.dim),
        v = zeros(Float32, p.dim),
        att = zeros(Float32, p.seq_len),
        logits = zeros(Float32, p.vocab_size),
        key_cache = zeros(Float32, p.dim, p.seq_len, p.n_layers),
        value_cache = zeros(Float32, p.dim, p.seq_len, p.n_layers))
end

@kwdef struct TokenizerConfig
    vocab::Vector{<:Vector{UInt8}}
    vocab_scores::Vector{Float32}
    max_token_length::Int
end
@kwdef struct TrainedModel
    config::Config
    weights::TransformerWeights
    tokenizer::TokenizerConfig
end
