# Llama2.jl

This project started as a port of Andrej Karpathy's llama2.c (https://github.com/karpathy/llama2.c).
The quantization code is a port from https://github.com/ggerganov/llama.cpp.

## Goals of this project

The goal is to provide code that is *readable* and *self-contained* (uses minimal dependencies).

## Installation

This package is not yet registered. It can be installed by running
```julia
pkg> add https://github.com/cafaxo/Llama2.jl
```

## Usage

We currently support two model formats:
- Andrej Karpathy's llama2.c format
- A subset of the GGML format (currently only the `q4_K_S` variant)

The llama2.c models can be found at https://huggingface.co/karpathy/tinyllamas.
With these models, the [tokenizer.bin](https://github.com/karpathy/llama2.c/raw/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin) file is also required.

Here is an output sample from the 42M tinyllama model:
```julia
julia> using Llama2

julia> model = load_karpathy_model("stories42M.bin", "tokenizer.bin");

julia> sample(model, "Tim was happy."; temperature = 0.8f0)
Tim was happy. He had a new toy. It was a big red car. He wanted to play with it all day.
Tim took his car outside. He found a drain. It was a big drain. Tim put his car on the drain. The car went down the drain.
Tim was sad. He missed his car. He went home. His mom saw him. She said, "Don't worry, we will get your car back." Tim was glad. He knew his mom would help him. They went to the drain. Tim's car came back. He was happy again.
-------
achieved tok/s: 282.80
```

A compatible Llama2 7B model can be downloaded from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML.

- **Only the q4_K_S variant is supported.**
- **Launch Julia with `julia -t auto` for better performance (multithreading)**

Here is an output sample:
```julia
julia> using Llama2

julia> model = load_ggml_model("llama-2-7b-chat.ggmlv3.q4_K_S.bin");

julia> sample(model, "The Julia programming language is")
The Julia programming language is an innovative language for technical computing and scientific research.
```
Thanks to weight quantization, a machine with 8GB RAM is sufficient to run this.
