using CUDA
using Llama2

model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B.Q4_K_S.gguf", AT=CuArray);
using Llama2: get_run_state
s = get_run_state(model)
w = model.weights.layers[1]
kv = s.kvcache_layers[1];

#%%
include("vecdot_opt.jl")

a = CUDA.randn(Float32, 1024*4 * 1024 *4)
sum_blocks_ka(a)
dev = KernelAbstractions.get_backend(a)
syncKA = KernelAbstractions.synchronize
@time (sum_blocks_ka(a), syncKA(dev))

sq_opt = similar(a)
@time (sum_blocks_ka(sq_opt), syncKA(dev))

num_blocks = cld(length(a), 32)
sums = similar(a, Float16, num_blocks)
@time (sum_blocks_ka!(sums, a), syncKA(dev))
@time (sum_blocks_ka!(sums, a), syncKA(dev))

;
#%%

4096 *1024* (2) / 0.000072 * 1e-9
#%%
4096 *1024*16* (2) / 0.001172 * 1e-9

#%%
using Llama2: to_block_f16_sums32_ka
r = (r = to_block_f16_sums32_ka(a); syncKA(dev); r)
@time (to_block_f16_sums32_ka(a), syncKA(dev))
@time (to_block_f16_sums32_ka(a), syncKA(dev));
#%%
@show size(r)
@show size(sums)
@show all(r .== sums)
@show r
@show sums
;
#%%
using Llama2: get_run_state
include("../cuda.jl_for_comparison/inference.CUDA.jl")

# @time @CUDA.sync matmul_cudajl!(s.q, w.wq, s.xb)
# @time @CUDA.sync matmul_cudajl!(s.q, w.wq, s.xb)
# @time @CUDA.sync matmul_cudajl!(s.q, w.wq, s.xb)
using Llama2: matmul!, dequantize!, rmsnorm!
using KernelAbstractions: get_backend
import KernelAbstractions
dev = KernelAbstractions.get_backend(s.q)
syncKA = KernelAbstractions.synchronize
token = 128001
dequantize!(s.x, model.weights.token_embedding_table[:, token]) 
rmsnorm!(s.xb, s.x, w.rms_att_weight)
@time (matmul!(s.q, w.wq, s.xb), syncKA(dev))
@time (matmul!(s.q, w.wq, s.xb), syncKA(dev))
# @display s.q
include("vecdot_opt.jl")
sq_opt = similar(s.q)
dev = KernelAbstractions.get_backend(sq_opt)
(matmul_opt!(sq_opt, w.wq, s.xb), syncKA(dev))
@time (matmul_opt!(sq_opt, w.wq, s.xb), syncKA(dev))
@time (matmul_opt!(sq_opt, w.wq, s.xb), syncKA(dev))
@show Array(sq_opt)[1:10]
condition = isapprox.(Array(s.q), Array(sq_opt), rtol=1e-2, atol=0.001)
@show findall(.!condition)[1:min(end,5)]
@assert all(condition)
;
#%%
include("vecdot.bench.jl")
sq_opt .= 0
@time (matmul!(s.q, w.wq, s.xb), syncKA(dev))
(matmul_v2!(sq_opt, w.wq, s.xb), syncKA(dev))
@time (matmul_v2!(sq_opt, w.wq, s.xb), syncKA(dev))
@time (matmul_v2!(sq_opt, w.wq, s.xb), syncKA(dev))
condition = isapprox.(Array(s.q), Array(sq_opt), rtol=1e-2, atol=0.0005)
@show Array(sq_opt)[1:10]
@show findall(.!condition)[1:min(end,5)]
@assert all(condition)
#%%
@typeof w.wq
@show Array(w.wq)[1,1].qs
#%%
using Llama2: rmsnorm!, combine_values!, attention_weights!, softmax_for!
(;dim,hidden_dim,n_layers,n_heads,n_kv_heads,) = model.config
head_size = dim ÷ n_heads
pos = 1

println("CUDA.jl rmsnorm:")
@time CUDA.@sync rmsnorm_cudajl!(s.xb, s.x, w.rms_att_weight)
@time CUDA.@sync rmsnorm_cudajl!(s.xb, s.x, w.rms_att_weight)
println("KA rmsnorm:")
@time rmsnorm!(s.xb, s.x, w.rms_att_weight), syncKA(dev)
@time rmsnorm!(s.xb, s.x, w.rms_att_weight), syncKA(dev)
#%%
println("CUDA.jl attention_weights:")
q = reshape(s.q, head_size, n_heads)
att = reshape(s.att[1:(n_heads*pos)], pos, n_heads)
@time CUDA.@sync attention_weights_cudajl!(att, kv.key_cache, q)
@time CUDA.@sync attention_weights_cudajl!(att, kv.key_cache, q);
println("KA attention_weights:")
@time attention_weights!(att, kv.key_cache, q), syncKA(dev)
@time attention_weights!(att, kv.key_cache, q), syncKA(dev)
;
#%%
println("CUDA.jl combine_values:")
xb = reshape(s.xb, head_size, n_heads)
@time CUDA.@sync combine_values_cudajl!(xb, kv.value_cache, att)
@time CUDA.@sync combine_values_cudajl!(xb, kv.value_cache, att)
println("KA combine_values:")
@time combine_values!(xb, kv.value_cache, att), syncKA(dev)
@time combine_values!(xb, kv.value_cache, att), syncKA(dev)
#%%
println("CUDA.jl softmax_for:")
@time softmax_for_cudajl!(att, n_heads)
@time softmax_for_cudajl!(att, n_heads)
println("KA softmax_for:")
@time softmax_for!(att, n_heads), syncKA(dev)
@time softmax_for!(att, n_heads), syncKA(dev)

;