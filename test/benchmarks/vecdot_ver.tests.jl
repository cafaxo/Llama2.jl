# INITIALIZATIONS for testing.
using CUDA
using Llama2

model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B.Q4_K_S.gguf", AT=CuArray);
using Llama2: get_run_state
s = get_run_state(model)
w = model.weights.layers[1]
kv = s.kvcache_layers[1];


import KernelAbstractions
dev = KernelAbstractions.get_backend(s.q)
syncKA = KernelAbstractions.synchronize
token = 128001

using Llama2: matmul!, dequantize!, rmsnorm!

dequantize!(s.x, model.weights.token_embedding_table[:, token]) 
rmsnorm!(s.xb, s.x, w.rms_att_weight)
include("vecdot.v1.jl")

#%% Testing runtime of different matmuls. q4
(matmul_v1!(s.q, w.wq, s.xb), syncKA(dev))
@time (matmul_v1!(s.q, w.wq, s.xb), syncKA(dev))
# @display @benchmark (matmul_v1!($s.q, $(w.wq), $(s.xb)), syncKA($dev))

s_q = similar(s.q)
(matmul!(s_q, w.wq, s.xb), syncKA(dev))
@time (matmul!(s_q, w.wq, s.xb), syncKA(dev))
@display @benchmark (matmul!($s_q, $(w.wq), $(s.xb)), syncKA($dev))
condition = isapprox.(Array(s.q), Array(s_q), rtol=1e-2, atol=0.0005)
@show findall(.!condition)[1:min(end,5)]
@assert all(condition)

#%% Testing q5 weight matmul with w.wv weight

# (matmul_v1!(s.v, w.wv, s.xb), syncKA(dev))
# @time (matmul_v1!(s.v, w.wv, s.xb), syncKA(dev))
# @display @benchmark (matmul_v1!($s.v, $(w.wv), $(s.xb)), syncKA($dev))

s_v = similar(s.v)
(matmul!(s_v, w.wv, s.xb), syncKA(dev))
@time (matmul!(s_v, w.wv, s.xb), syncKA(dev))
@display @benchmark (matmul!($s_v, $(w.wv), $(s.xb)), syncKA($dev))

#%% Testing q6 weight by .output_weight weight
s.logits .= 0
(matmul_v1!(s.logits, model.weights.output_weight, s.x), syncKA(dev))
@time (matmul_v1!(s.logits, model.weights.output_weight, s.x), syncKA(dev))
# @display @benchmark (matmul_v1!($s.logits, $(model.weights.output_weight), $(s.x)), syncKA($dev))

s_logits = similar(s.logits)
s_logits .= 0
(matmul!(s_logits, model.weights.output_weight, s.x), syncKA(dev))
@time (matmul!(s_logits, model.weights.output_weight, s.x), syncKA(dev))
@display @benchmark (matmul!($s_logits, $(model.weights.output_weight), $(s.x)), syncKA($dev))
condition = isapprox.(Array(s.logits), Array(s_logits), rtol=1e-2, atol=0.0005)
# show first 5 element
@show Array(s.logits)[1:5]
@show Array(s_logits)[1:5]  
@show findall(.!condition)[1:min(end,5)]
@assert all(condition)
@show size(model.weights.output_weight)
#%% Check w1 matmul
s_hb = similar(s.hb)
(matmul!(s_hb, w.w1, s.xb), syncKA(dev))
@time (matmul!(s_hb, w.w1, s.xb), syncKA(dev))
@display @benchmark (matmul!($s_hb, $(w.w1), $(s.xb)), syncKA($dev))


