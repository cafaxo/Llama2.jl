using Boilerplate
using CUDA
import CUDA.CuArray  # Add this import statement

@kwdef struct RunStateCUDA
  # current wave of activations
  x::CuVector{Float32}      # activation at current time stamp (dim,)
  xb::CuVector{Float32}     # same, but inside a residual branch (dim,)
  xb2::CuVector{Float32}    # an additional buffer just for convenience (dim,)
  hb::CuVector{Float32}     # buffer for hidden dimension in the ffn (hidden_dim,)
  hb2::CuVector{Float32}    # buffer for hidden dimension in the ffn (hidden_dim,)
  q::CuVector{Float32}      # query (dim,)
  k::CuVector{Float32}      # key (dim,)
  v::CuVector{Float32}      # value (dim,)
  att::CuVector{Float32}    # buffer for scores/attention values (seq_len * n_heads,)
  logits::CuVector{Float32} # output logits
  # kv cache
  kvcache_layers::Vector{KVCache}
end
RunStateCUDA(rs::RunState) = RunStateCUDA(
  x              = cu(rs.x),
  xb             = cu(rs.xb),
  xb2            = cu(rs.xb2),
  hb             = cu(rs.hb),
  hb2            = cu(rs.hb2),
  q              = cu(rs.q),
  k              = cu(rs.k),
  v              = cu(rs.v),
  att            = cu(rs.att),
  logits         = cu(rs.logits),
  kvcache_layers = [KVCache(cu(kv.key_cache), cu(kv.value_cache)) for kv in rs.kvcache_layers],
)

@kwdef struct TransformerLayerWeightsCUDA{T, T2}
    # weights for rmsnorms
    rms_att_weight::CuArray{Float32, 1} # (dim,)
    rms_ffn_weight::CuArray{Float32, 1} # (dim,)
    # weights for matmuls
    wq::CuArray{T, 2} # (dim, dim)
    wk::CuArray{T, 2} # (dim, dim)
    wv::CuArray{T2, 2} # (dim, dim)
    wo::CuArray{T, 2} # (dim, dim)
    # weights for ffn
    w1::CuArray{T, 2} # (dim, hidden_dim)
    w2::CuArray{T2, 2} # (hidden_dim, dim)
    w3::CuArray{T, 2} # (dim, hidden_dim)
end
using Boilerplate
function to_cuda(weights::TransformerLayerWeights)
    return TransformerLayerWeightsCUDA(
        rms_att_weight = cu(weights.rms_att_weight),
        rms_ffn_weight = cu(weights.rms_ffn_weight),
        wq = cu(weights.wq),
        wk = cu(weights.wk),
        wv = cu(weights.wv),
        wo = cu(weights.wo),
        w1 = cu(weights.w1),
        w2 = cu(weights.w2),
        w3 = cu(weights.w3)
    )
end
@kwdef struct TransformerWeightsCUDA
    token_embedding_table::CuArray # (dim, vocab_size)
    layers::Vector{TransformerLayerWeightsCUDA}
    # final rmsnorm
    rms_final_weight::CuVector{Float32} # (dim,)
    output_weight::CuArray # (dim, vocab_size)
end
function to_cuda(weights::TransformerWeights)
    return TransformerWeightsCUDA(
        token_embedding_table = cu(weights.token_embedding_table),
        layers = [to_cuda(layer) for layer in weights.layers],
        rms_final_weight = cu(weights.rms_final_weight),
        output_weight = cu(weights.output_weight)
    )
end

struct LanguageModelCUDA{TOK<:Tokenizer}
  config::ModelConfig
  tokenizer::TOK
  weights::TransformerWeightsCUDA
end
function to_cuda(llm::LanguageModel)
  return LanguageModelCUDA(
      llm.config, llm.tokenizer, to_cuda(llm.weights)
  )
end

@views function transformer!(token::Int, pos::Int, config::ModelConfig, s::RunStateCUDA, weights::TransformerWeightsCUDA)
  x = s.x

  (;
      dim,
      hidden_dim,
      n_layers,
      n_heads,
  ) = config

  head_size = dim ÷ n_heads
  CUDA.allowscalar(true)

  @typeof weights.token_embedding_table
  @typeof x
  @typeof token
  # copy the token embedding into x
  @time dequantize!(Array(x), Array(weights.token_embedding_table[:, token]))
  @time dequantize!(x, weights.token_embedding_table[:, token])
  @sizes x
  @sizes weights.token_embedding_table[:, token]
#   @show weights.token_embedding_table[:, token][1]

  # forward all the layers
  for l in 1:n_layers
      w = weights.layers[l]
      kv = s.kvcache_layers[l]

      # attention rmsnorm
      rmsnorm!(s.xb, x, w.rms_att_weight)

      # qkv matmuls for this position
      matmul_res = Array(s.q)
      matmul!(matmul_res, Array(w.wq), Array(s.xb))
      @sizes matmul_res
      @show matmul_res[1:10]
      matmul_A = zeros(Float32, size(w.wq, 1)*256 * size(w.wq, 2))
      dequantize!(matmul_A , reshape(Array(w.wq), :)) # just for the test. to call the basic matmul
      matmul_A  = reshape(matmul_A , size(w.wq, 1)*256, size(w.wq, 2))
      matmul_res2 = Array(s.q)
      matmul!(matmul_res2, matmul_A, Array(s.xb))

      @show matmul_res2[1:10]
      @sizes s.xb
      @sizes Array(w.wq)
      matmul!(s.q, w.wq, s.xb)
      @sizes s.q
      @show s.q[1:10]
      @assert false "OK"
      matmul!(s.k, w.wk, s.xb)
      matmul!(s.v, w.wv, s.xb)

      q = reshape(s.q, head_size, n_heads)
      k = reshape(s.k, head_size, n_heads)

      # apply RoPE rotation to the q and k vectors for each head
      rope!(q, pos)
      rope!(k, pos)

      # save key,value at this time step (pos) to our kv cache
      copyto!(kv.key_cache[:, :, pos], s.k)
      copyto!(kv.value_cache[pos, :, :], s.v)

      # take a contiguous slice of the attention buffer
      att = reshape(s.att[1:(n_heads*pos)], pos, n_heads)

      # multihead attention
      attention_weights!(att, kv.key_cache, q)

      att ./= sqrt(Float32(head_size))

      for h in 1:n_heads
          # softmax the scores to get attention weights
          softmax!(att[:, h])
      end

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
