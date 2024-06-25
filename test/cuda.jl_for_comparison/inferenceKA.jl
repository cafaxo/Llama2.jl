
struct KVCacheCUDA
  key_cache::CuArray{Float32,3}   # (head_size, n_heads, seq_len)
  value_cache::CuArray{Float32,3} # (seq_len, head_size, n_heads)
end
CUDA.cu(kv::KVCache) = KVCacheCUDA(cu(kv.key_cache), cu(kv.value_cache))

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
  kvcache_layers::Vector{KVCacheCUDA}
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
  kvcache_layers = [cu(kv) for kv in rs.kvcache_layers],
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
function to_cuda(weights::TransformerLayerWeights)
    return TransformerLayerWeights{CuArray}(
        rms_att_weight = cu(weights.rms_att_weight),
        rms_ffn_weight = cu(weights.rms_ffn_weight),
        wq = cu(weights.wq),
        wk = cu(weights.wk),
        wv = cu(weights.wv),
        wo = cu(weights.wo),
        w1 = cu(weights.w1),
        w2 = cu(weights.w2),
        w3 = cu(weights.w3),
    )
end
@kwdef struct TransformerWeightsCUDA
    token_embedding_table::CuArray # (dim, vocab_size)
    layers::Vector{TransformerLayerWeights}
    # final rmsnorm
    rms_final_weight::CuVector{Float32} # (dim,)
    output_weight::CuArray # (dim, vocab_size)
end

function to_cuda(weights::TransformerWeights{T}) where T <: AbstractArray
    return TransformerWeights{CuArray}(
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
get_run_state(model::LanguageModelCUDA) = RunStateCUDA(RunState(model.config))
function to_cuda(llm::LanguageModel{Array})
  return LanguageModelCUDA(
      llm.config, llm.tokenizer, to_cuda(llm.weights)
  )
end

# ChatGPT-4o generated rope! without the ComplexF32 reinterpret and cis written out.
function rope_kernel(x, pos, head_size_div2, n_heads, theta_scale, freq_scale)
    head = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if head <= n_heads
        theta = freq_scale * (pos - 1)
        
        for i in 1:head_size_div2
            real_part = x[2 * (i - 1) + 1, head]
            imag_part = x[2 * (i - 1) + 2, head]
            c = cos(theta)
            s = sin(theta)

            new_real = real_part * c - imag_part * s
            new_imag = real_part * s + imag_part * c
            
            x[2 * (i - 1) + 1, head] = new_real
            x[2 * (i - 1) + 2, head] = new_imag
            
            theta *= theta_scale
        end
    end
    nothing
end

function rope!(x::CuMatrix{Float32}, pos::Int, config::ModelConfig)
  head_size, n_heads = size(x)
  head_size_div2 = head_size ÷ 2
  freq_base = config.rope_freq_base
  freq_scale = 1.0f0

  theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))

  threads_per_block = 256
  blocks_per_grid = ceil(Int, n_heads / threads_per_block)

  @cuda threads=threads_per_block blocks=blocks_per_grid rope_kernel(x, pos, head_size_div2, n_heads, theta_scale, freq_scale)

  return nothing
end
# Sonnet 3.5 generated attention_weights from the Llama2.jl CPU code.
function attention_weights_kernel!(att, key_cache, q, n_gqa)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    h = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if t <= size(att, 1) && h <= size(att, 2)
        key_h = (h - 1) ÷ n_gqa + 1
        s = 0f0

        @inbounds for j in 1:size(q, 1)
            s += q[j, h] * key_cache[j, key_h, t]
            # h == 3 && @cushow q[j, h]
            # h == 3 && @cushow key_cache[j, key_h, t]
        end
        @inbounds att[t, h] = s
    end

    return nothing
end

function attention_weights!(att::AbstractArray, key_cache::AbstractArray, q::AbstractArray)
    n_gqa = size(q, 2) ÷ size(key_cache, 2)

    threads_per_block = (16, 16)
    blocks_per_grid = (
      cld(size(att, 1), threads_per_block[1]),
      cld(size(att, 2), threads_per_block[2])
    )

    @cuda threads=threads_per_block blocks=blocks_per_grid attention_weights_kernel!(att, key_cache, q, n_gqa)

    return att
end

# chatGPT-4o generated combine_values from the Llama2.jl CPU code.
function combine_values_kernel(xb, value_cache, att, n_gqa)
  i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  h = threadIdx().y + (blockIdx().y - 1) * blockDim().y
  
  if i <= size(xb, 1) && h <= size(xb, 2)
      s = 0.0f0
      value_h = 1 + div(h - 1, n_gqa)
      
      for t in 1:size(att, 1)
          s += att[t, h] * value_cache[t, i, value_h]
      end
      
      xb[i, h] = s
  end
  return nothing
end
function combine_values!(xb::AbstractMatrix, value_cache::AbstractArray, att) where T
  n_gqa = size(att, 2) ÷ size(value_cache, 3)
  
  threads_per_block = (16, 16)
  blocks_per_grid = (ceil(Int, size(xb, 1) / threads_per_block[1]), ceil(Int, size(xb, 2) / threads_per_block[2]))

  @cuda threads=threads_per_block blocks=blocks_per_grid combine_values_kernel(xb, value_cache, att, n_gqa)
end

function softmax_kernel(att, attention_maximum)
    h = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if h <= size(att, 2)
        max_val = attention_maximum[h]

        exp_sum = 0.0f0
        for t in 1:size(att, 1)
            exp_val = exp(att[t, h] - max_val)
            exp_sum += exp_val
            att[t, h] = exp_val
        end

        for t in 1:size(att, 1)
            att[t, h] /= exp_sum
        end
    end
end
@views function softmax_for!(att::AbstractMatrix, n_heads::Int)
  # n_heads = size(att, 2)

  threads_per_block = 256
  blocks_per_grid = ceil(Int, n_heads / threads_per_block)

  att_max = reshape(maximum(att, dims=1), 1, :)
  # CUDA.maximum!(att_max, att, dims=1)

  @cuda threads=threads_per_block blocks=blocks_per_grid softmax_kernel(att, att_max)
end

# @views function softmax_for!(att::CuMatrix{Float32}, n_heads::Int)

#   threads_per_block = (16, 16)
#   blocks_per_grid = (ceil(Int, size(att, 1) / threads_per_block[1]), ceil(Int, n_heads / threads_per_block[2]))
#   att_max = reshape(maximum(att, dims=1), 1, :)
#   @cuda threads=threads_per_block blocks=blocks_per_grid softmax_kernel(att, att_max, n_heads)
# end
# Llama70B generated rmsnorm from the Llama2.jl CPU code.
function rmsnorm_kernel(o, x, weight, length_x)
    # Calculate thread index
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i <= length_x
        # Step 1: Calculate the sum of squares
        ss = 0.0f0
        for j in 1:length_x
            ss += x[j] * x[j]
        end

        # Step 2: Normalize and scale
        ss /= length_x
        ss += 1f-6
        ss = 1f0 / sqrt(ss)
        o[i] = weight[i] * (ss * x[i])
    end
    nothing
end
function rmsnorm!(o::AbstractVector, x::AbstractVector, weight::AbstractVector)
  length_x = length(x)

  threads_per_block = 256
  blocks_per_grid = ceil(Int, length_x / threads_per_block)

  @cuda threads=threads_per_block blocks=blocks_per_grid rmsnorm_kernel(o, x, weight, length_x)
end


@views function transformer!(token::Int, pos::Int, config::ModelConfig, s::RunState{CuArray}, weights::TransformerWeights{CuArray})
  x = s.x

  (;
      dim,
      hidden_dim,
      n_layers,
      n_heads,
      n_kv_heads,
  ) = config

  head_size = dim ÷ n_heads
  CUDA.@sync begin # syncing for safety reasons, to make sure that step timings are not measured incorrectly from the outside.
  # copy the token embedding into x
  dequantize_cuda!(x, weights.token_embedding_table[:, token])

  # forward all the layers
  for l in 1:n_layers
      w = weights.layers[l]
      kv = s.kvcache_layers[l]

      # attention rmsnorm
      @time CUDA.@sync rmsnorm!(s.xb, x, w.rms_att_weight)

      # qkv matmuls for this position
      @time CUDA.@sync matmul!(s.q, w.wq, s.xb)
      matmul!(s.k, w.wk, s.xb)
      matmul!(s.v, w.wv, s.xb)
      
      q = reshape(s.q, head_size, n_heads)
      k = reshape(s.k, head_size, n_kv_heads)
      
      # apply RoPE rotation to the q and k vectors for each head
      @time CUDA.@sync rope!(q, pos, config.rope_freq_base)
      rope!(k, pos, config.rope_freq_base)

      # save key,value at this time step (pos) to our kv cache
      @time CUDA.@sync copyto!(kv.key_cache[:, :, pos], s.k)
      copyto!(kv.value_cache[pos, :, :], s.v)

      # take a contiguous slice of the attention buffer
      att = reshape(s.att[1:(n_heads*pos)], pos, n_heads)

      # multihead attention
      @time CUDA.@sync attention_weights!(att, kv.key_cache, q)

      att ./= sqrt(Float32(head_size))

      softmax_for!(att, n_heads)

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
  
  end
  return nothing
end