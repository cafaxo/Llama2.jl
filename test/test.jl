using Llama2

using CUDA
# Options from KA: CuArray, Array, ROCArray, oneArray, MtlArray
model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B.Q4_K_S.gguf", AT=CuArray);

#%%
# GPU sample
res = sample(model, "Tim was happy."; temperature = 0.0f0, max_seq_len=10)
# res = sample(model, "Write me a simple is_prime function in julia"; temperature = 0.0f0, max_seq_len=400)
# res = sample(model, "Here is a simple implementation of an `is_prime` function in Julia:"; temperature = 0.0f0, max_seq_len=400)
# res = sample(model, "What is 3333+777? The answer is"; temperature = 0.0f0, max_seq_len=50)
# res = sample(model, "3333+777?"; temperature = 0.0f0, max_seq_len=20)
# res = sample(model, "3333+777?"; temperature = 0.0f0, max_seq_len=20)
# res = sample(model, "Write me a simple is_prime function in julia:"; temperature = 0.0f0, max_seq_len=400)

# CPU sample for debugging model
# res = sample(model, "Tim was happy."; temperature = 0.0f0, max_seq_len=10)

