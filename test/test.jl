using Llama2

model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B.Q4_K_S.gguf");

#%%
using CUDA
using Llama2: to_cuda
model_cu = nothing
# we need to free cuda memory before loading a new model
CUDA.reclaim(); GC.gc(true); #CUDA.memory_status()
model_cu = to_cuda(model);

#%%
# GPU sample
res = sample(model_cu, "Tim was happy."; temperature = 0.0f0, max_seq_len=10)
# res = sample(model_cu, "Write me a simple is_prime function in julia"; temperature = 0.0f0, max_seq_len=400)
# res = sample(model_cu, "Here is a simple implementation of an `is_prime` function in Julia:"; temperature = 0.0f0, max_seq_len=400)
# res = sample(model_cu, "What is 3333+777? The answer is"; temperature = 0.0f0, max_seq_len=50)
# res = sample(model_cu, "3333+777?"; temperature = 0.0f0, max_seq_len=20)
# res = sample(model_cu, "3333+777?"; temperature = 0.0f0, max_seq_len=20)
# res = sample(model_cu, "Write me a simple is_prime function in julia:"; temperature = 0.0f0, max_seq_len=400)

# CPU sample for debugging model
# res = sample(model, "Tim was happy."; temperature = 0.0f0, max_seq_len=10)

