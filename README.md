# Llama2.jl

Llama2.jl supports inference and training of Llama-style language models.

## Installation

This package is not yet registered. It can be installed by running
```julia
pkg> add https://github.com/cafaxo/Llama2.jl
```

## Usage

We currently support:
- A subset of the GGUF format
- Andrej Karpathy's llama2.c format

Experimental Llama 3 support is available:

```julia
julia> model = load_gguf_model("Meta-Llama-3-8B.Q4_K_S.gguf")
LanguageModel(
ModelConfig(
  dim            = 4096,
  hidden_dim     = 14336,
  n_layers       = 32,
  n_heads        = 32,
  n_kv_heads     = 8,
  vocab_size     = 128256,
  seq_len        = 512,
  rope_freq_base = 500000.0,
))

julia> sample(model, "The Julia programming language is"; temperature=0.0f0)
 The Julia programming language is a high-level, high-performance dynamic language for technical computing, with syntax that is familiar to users of other technical computing environments. It provides a sophisticated compiler, distributed parallel execution, numerical accuracy, and MATLABÂ®- and R-compatibility. It is open-source and available on 32- and 64-bit x86-based operating systems (Windows, Linux, and Mac OS X). Julia is sponsored by the Julia Computing Inc. company.
```

Andrej Karpathy's llama2.c models can be found at https://huggingface.co/karpathy/tinyllamas.
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

## Experimental training support

Llama2.jl can train very small Llama-style models on the CPU:
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
