# Llama2.jl

Llama2.jl provides simple code for inference and training of llama2-based language models.

## Installation

This package is not yet registered. It can be installed by running
```julia
pkg> add https://github.com/cafaxo/Llama2.jl
```

## Usage

We currently support two model formats:
- Andrej Karpathy's llama2.c format
- A subset of the GGUF format (currently only the `q4_K_S` variant of Llama 2 models are tested)

The llama2.c models can be found at https://huggingface.co/karpathy/tinyllamas.
With these models, the [tokenizer.bin](https://github.com/karpathy/llama2.c/raw/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin) file is also required.

Here is an output sample from the 42M tinyllama model:
```julia
julia> using Llama2

julia> download("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin", "stories42M.bin")
"stories42M.bin"

julia> download("https://raw.githubusercontent.com/karpathy/llama2.c/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin", "tokenizer.bin")
"tokenizer.bin"

julia> model = load_karpathy_model("stories42M.bin", "tokenizer.bin");

julia> sample(model, "Tim was happy."; temperature = 0.8f0)
Tim was happy. He had a new toy. It was a big red car. He wanted to play with it all day.
Tim took his car outside. He found a drain. It was a big drain. Tim put his car on the drain. The car went down the drain.
Tim was sad. He missed his car. He went home. His mom saw him. She said, "Don't worry, we will get your car back." Tim was glad. He knew his mom would help him. They went to the drain. Tim's car came back. He was happy again.
-------
achieved tok/s: 282.80
```

A compatible Llama2 7B model can be downloaded from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF.

- **Only the q4_K_S variant is tested.**
- **Launch Julia with `julia -t auto` for better performance (multithreading)**

Here is an output sample:
```julia
julia> using Llama2

julia> model = load_gguf_model("llama-2-7b-chat.Q4_K_S.gguf");

julia> sample(model, "The Julia programming language is")
The Julia programming language is an innovative language for technical computing and scientific research.
```
Thanks to weight quantization, a machine with 8GB RAM is sufficient to run this.

## Experimental training support

Llama2.jl can now train tiny llama2 models on the CPU:
```julia
julia> using Llama2

julia> text = read("tinyshakespeare.txt", String);

julia> tokenizer = CharTokenizer(text);

julia> tokens = encode(text, tokenizer);

julia> config = ModelConfig(dim=64, hidden_dim=96, n_layers=4, n_heads=4, n_kv_heads=4, vocab_size=length(tokenizer.id_to_token), seq_len=128)
ModelConfig(
  dim         = 64,
  hidden_dim  = 96,
  n_layers    = 4,
  n_heads     = 4,
  n_kv_heads  = 4,
  vocab_size  = 65,
  seq_len     = 128,
)

julia> weights = train(config, tokens; n_tokens=4_000_000, batch_size=4);
Training a model with 148160 parameters...
Progress: 100%|███████████████████████████| Time: 0:01:31 (11.66 ms/it)
  iteration:      7812 / 7812
  training_loss:  1.497

julia> model = LanguageModel(config, tokenizer, weights);

julia> sample(model, "Julia is"; stop_on_special_token=false, bos_token=false)
Julia is to seen all?

FLORIZEL:
Now the sir?

JOHN ORTHNROLIO:
Talksabated with me with a more thou Vrequitest.
The city good of
-------
achieved tok/s: 11479.52
```

## Acknowledgements

This project started as a port of Andrej Karpathy's llama2.c (https://github.com/karpathy/llama2.c).
The quantization code is a port from https://github.com/ggerganov/llama.cpp.
