using Llama2

# https://huggingface.co/shaowenchen/chinese-alpaca-2-7b-16k-gguf/tree/main
# file_name = "/home/zhangyong/codes/julia_learn/others/llm/models/llama2/chinese-alpaca-2-7b-16k.Q4_K_S.gguf"
# https://huggingface.co/shaowenchen/chinese-llama-2-7b-16k-gguf/tree/main
file_name = "/home/zhangyong/codes/julia_learn/others/llm/models/llama2/chinese-llama-2-7b-16k.Q4_K_S.gguf"

# https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/tree/main
file_name = "/home/zhangyong/codes/julia_learn/others/llm/models/llama2/llama-2-7b-chat.Q4_K_S.gguf"
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
# file_name = "/home/zhangyong/codes/julia_learn/others/llm/models/llama2/llama-2-7b-chat.ggmlv3.q4_K_S.bin"

# https://huggingface.co/about0/llama-v2-chinese-ymcui-alpaca-GGML-13B/blob/main/llama-v2-chinese-alpaca-13B-Q4_K_S.ggml
# file_name = "/home/zhangyong/codes/julia_learn/others/llm/models/llama2/llama-v2-chinese-alpaca-13B-Q4_K_S.ggml"  # ok, 可用. 8.3G内存

file_name = "/home/zhangyong/codes/julia_learn/others/llm/models/llama2/sciphi-self-rag-mistral-7b-32k.Q4_K_S.gguf"  # ok, 可用. 8.3G内存


model = load_gguf_model(file_name)
# model = load_ggml_model(file_name)
# sample(model, "The Julia programming language is"; temperature = 1.0f0)
# sample(model, "What are the prime numbers up to 100?")  # ; temperature = 0.8f0
# sample(model, "The capital of China is"; temperature = 1.0f0)  # The capital of China is 中国的首都是
sample(model, "China"; temperature = 1.0f0)

# julia -t auto --project=. src/test_1.jl
# 测试的中文聊天效果很差劲.没用.  2023.10.25
