using Llama2

# model = load_gguf_model("/llms/gguf/dolphin-2.6-mistral-7b.Q5_K_M.gguf");
# model = load_gguf_model("/llms/gguf/openhermes-2-mistral-7b.Q4_K_M.gguf");
# model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf");
# model = load_gguf_model("/llms/gguf/llama-2-7b-chat.Q4_K_S.gguf");
model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B.Q4_K_S.gguf");
#%%
using CUDA
using Llama2: to_cuda
model_cu = nothing
# we need to free cuda memory before loading a new model
CUDA.reclaim(); GC.gc(true); #CUDA.memory_status()
model_cu = to_cuda(model);
#%%
# res = sample(model, "Write me a simple is_prime function in julia"; temperature = 0.0f0, max_seq_len=400)
# res = sample(model, "Here is a simple implementation of an `is_prime` function in Julia:"; temperature = 0.0f0, max_seq_len=400)
# res = sample(model, "What is 3333+777? The answer is"; temperature = 0.0f0, max_seq_len=50)
# res = sample(model, "3333+777?"; temperature = 0.0f0, max_seq_len=20)
res = sample(model_cu, "Tim was happy."; temperature = 0.0f0, max_seq_len=6)
@show res

