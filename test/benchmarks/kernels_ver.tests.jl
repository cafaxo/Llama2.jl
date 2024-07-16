using CUDA
using Llama2

model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B.Q4_K_S.gguf", array_type=CuArray);
using Llama2: get_run_state
s = get_run_state(model)
w = model.weights.layers[1]
kv = s.kvcache_layers[1];

(;dim, hidden_dim, n_layers, n_heads, n_kv_heads,rope_freq_base) = model.config
head_size = dim ÷ n_heads

import KernelAbstractions
dev = KernelAbstractions.get_backend(s.q)
syncKA = KernelAbstractions.synchronize
token = 128001

using Llama2: matmul!, dequantize!, rmsnorm!

dequantize!(s.x, model.weights.token_embedding_table[:, token]) 
rmsnorm!(s.xb, s.x, w.rms_att_weight)
include("kernels.v1.jl")

#%%
(rmsnorm_v1!(s.xb, s.x, w.rms_att_weight), syncKA(dev))
@time (rmsnorm_v1!(s.xb, s.x, w.rms_att_weight), syncKA(dev)) 
# @display @benchmark (rmsnorm_v1!($(s.xb), $(s.x), $(w.rms_att_weight)), syncKA($dev))

s_xb = zero(s.xb)
(rmsnorm!(s_xb, s.x, w.rms_att_weight), syncKA(dev))
@time (rmsnorm!(s_xb, s.x, w.rms_att_weight), syncKA(dev)) 
@display @benchmark (rmsnorm!($(s_xb), $(s.x), $(w.rms_att_weight)), syncKA($dev))
condition = isapprox.(Array(s.xb), Array(s_xb), rtol=1e-2, atol=0.0005)
@show Array(s.xb)[250:260]
@show Array(s_xb)[250:260]
@show findall(.!condition)[1:min(end,5)]
@assert all(condition)

#%% rope speedbenchmarking
using Llama2: rope!
position_max = 300
pos = 300

matmul!(s.q, w.wq, s.xb)
matmul!(s.k, w.wk, s.xb)
q = reshape(s.q, head_size, n_heads)
k = reshape(s.k, head_size, n_kv_heads)
q_orig = deepcopy(q)
(rope_v1!(q, pos, rope_freq_base), syncKA(dev))
# @display @benchmark (rope_v1!($(q), $(pos), $(rope_freq_base)), syncKA($dev))
q = deepcopy(q_orig)
@time (rope_v1!(q, pos, rope_freq_base), syncKA(dev)) 
q2 = deepcopy(q_orig)
(rope!(q2, pos, rope_freq_base), syncKA(dev))
@display @benchmark (rope!($(q2), $(pos), $(rope_freq_base)), syncKA($dev))
q2 = deepcopy(q_orig)
@time (rope!(q2, pos, rope_freq_base), syncKA(dev)) 

# @show q[1:10]
# @show q2[1:10]
cond = isapprox.(Array(q), Array(q2), rtol=1e-2, atol=0.0005)
@show findall(.!cond)[1:min(end,5)]
@assert all(cond)

#%% copyto! benchmarking
@views for p in 1:position_max
  copyto!(kv.key_cache[:, :, p], CUDA.randn(Float32, size(s.k)))
  copyto!(kv.value_cache[p, :, :], CUDA.randn(Float32, size(s.v)))
end
@sizes kv.value_cache
@sizes kv.key_cache
#%% attention_weights! benchmarking
using Llama2: attention_weights!

att = reshape(s.att[1:(n_heads*pos)], pos, n_heads)
att ./= sqrt(Float32(head_size))
orig_att = deepcopy(att)

attention_weights_v1!(att, kv.key_cache, q)

(attention_weights_v1!(att, kv.key_cache, q), syncKA(dev))
# @display @benchmark (attention_weights_v1!($(att), $(kv.key_cache), $(q)), syncKA($dev))
att .= orig_att
@time (attention_weights_v1!(att, kv.key_cache, q), syncKA(dev)) 

att2 = deepcopy(orig_att)
(attention_weights!(att2, kv.key_cache, q), syncKA(dev))
@display @benchmark (attention_weights!($(att2), $(kv.key_cache), $(q)), syncKA($dev))
att2 .= orig_att
@time (attention_weights!(att2, kv.key_cache, q), syncKA(dev)) 


@show att[1:3,1:3]
@show att2[1:3,1:3]
cond = isapprox.(Array(att), Array(att2), rtol=1e-2, atol=0.0005)
@show findall(.!cond)[1:min(end,5)]
@assert all(cond)

#%% softmax_for! benchmarking
using Llama2: softmax_for!, attention_weights!
att = reshape(s.att[1:(n_heads*pos)], pos, n_heads)
# multihead attention
attention_weights!(att, kv.key_cache, q)
att ./= sqrt(Float32(head_size))

orig_att = deepcopy(att)

(softmax_for_v1!(att, n_heads), syncKA(dev))
# @display @benchmark (softmax_for_v1!($(att), $(n_heads)), syncKA($dev))
att .= orig_att
@time (softmax_for_v1!(att, n_heads), syncKA(dev)) 
(softmax_for!(att2), syncKA(dev))
@display @benchmark (softmax_for!($(att2)), syncKA($dev))
att2 .= orig_att
@time (softmax_for!(att2), syncKA(dev)) 

cond = isapprox.(Array(att), Array(att2), rtol=1e-2, atol=0.0005)
@show att[1:10, 1:3]
@show att2[1:10,1:3]
@show findall(.!cond)[1:min(end,5)]
@assert all(cond)


#%%
@sizes att
@display @benchmark maximum($att, dims=1)

#%% combine_values! benchmarking
using Llama2: combine_values!

xb = reshape(s.xb, head_size, n_heads)
(combine_values_v1!(xb, kv.value_cache, att), syncKA(dev))
@time (combine_values_v1!(xb, kv.value_cache, att), syncKA(dev)) 
@display @benchmark (combine_values_v1!($(xb), $(kv.value_cache), $(att)), syncKA($dev))

#%% benchmark the rest:
matmul!(s.xb2, w.wo, s.xb)
@display @benchmark (matmul!($(s.xb2), $(w.wo), $(s.xb)), syncKA($dev))
rmsnorm!(s.xb, s.x, w.rms_ffn_weight)
@display @benchmark (rmsnorm!($(s.xb), $(s.x), $(w.rms_ffn_weight)), syncKA($dev))

# Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
# first calculate self.w1(x) and self.w3(x)
@display @benchmark (matmul!($s.hb, $w.w1, $s.xb), syncKA($dev))
@display @benchmark (matmul!($s.hb2, $w.w3, $s.xb), syncKA($dev))

using Llama2: silu
# F.silu silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
s.hb .= silu.(s.hb)
s.hb .*= s.hb2
;
#%%
@typeof w.w1
@sizes Array(w.w1)
@typeof w.w3
@sizes Array(w.w3)
#%%
# final matmul to get the output of the ffn
@typeof w.w2
@sizes Array(w.w2)
@display @benchmark (matmul!($s.xb, $w.w2, $s.hb), syncKA($dev))
#%%
@display @benchmark begin
  combine_values_v1!($(xb), $(kv.value_cache), $(att))
  matmul!($(s.xb2), $(w.wo), $(s.xb))
  rmsnorm!($s.xb, $s.x, $w.rms_att_weight)

  # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
  # first calculate self.w1(x) and self.w3(x)
  matmul!($s.hb2, $w.w3, $s.xb)
  matmul!($s.hb, $w.w1, $s.xb)
  # final matmul to get the output of the ffn
  matmul!($s.xb, $w.w2, $s.hb)
  syncKA($dev)
end
#%%


copyto!(kv.key_cache[:, :, pos], s.k)
@time (copyto!(kv.key_cache[:, :, pos], s.k), syncKA(dev)) 
@display @benchmark (copyto!($(kv.key_cache[:, :, pos]), $(s.k)), syncKA($dev))









