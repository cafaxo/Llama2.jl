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
