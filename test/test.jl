#%%
using Llama2
download("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin", "stories42M.bin")
download("https://raw.githubusercontent.com/karpathy/llama2.c/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin", "tokenizer.bin")
"tokenizer.bin"
#%%
using Llama2

model = load_karpathy_model("stories42M.bin", "tokenizer.bin");

sample(model, "Tim was happy."; temperature = 0.0f0, bos_token=true)
#%%

using Llama2

# model = load_gguf_model("/llms/gguf/dolphin-2.6-mistral-7b.Q5_K_M.gguf");
# model = load_gguf_model("/llms/gguf/openhermes-2-mistral-7b.Q4_K_M.gguf");
# model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf");
# model = load_gguf_model("/llms/gguf/llama-2-7b-chat.Q4_K_S.gguf");
model = load_gguf_model("/llms/gguf/Meta-Llama-3-8B.Q4_K_S.gguf");
#%%
model_cu = to_cuda(model)
;
#%%
isqrt(11)
#%%
# res = sample(model, "Write me a simple is_prime function in julia"; temperature = 0.0f0, max_seq_len=400)
# res = sample(model, "Here is a simple implementation of an `is_prime` function in Julia:"; temperature = 0.0f0, max_seq_len=400)
# res = sample(model, "What is 3333+777? The answer is"; temperature = 0.0f0, max_seq_len=50)
res = sample(model, "3333+777?"; temperature = 0.0f0, max_seq_len=20)
# res = sample(model_cu, "Tim was happy."; temperature = 0.0f0, max_seq_len=6)
@show res

