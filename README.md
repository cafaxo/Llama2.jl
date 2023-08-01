# Llama2.jl

This project started as a port of Andrej Karpathy's llama2.c (https://github.com/karpathy/llama2.c).
The quantization code is a port from https://github.com/ggerganov/llama.cpp.

## Goals of this project

The goal is to provide code that is *readable* and *self-contained* (uses minimal dependencies).

## Installation

This package is not yet registered. It can be installed by running
```
pkg> add https://github.com/cafaxo/Llama2.jl
```

## Usage

We currently support two model formats:
- Andrej Karpathy's llama2.c format
- A subset of the GGML format (currently only the `q4_K_S` variant)

Download one of the llama2.c models from https://huggingface.co/karpathy/tinyllamas and the [tokenizer.bin](https://github.com/karpathy/llama2.c/raw/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin) file.

The model can be sampled with
```
julia> using Llama2

julia> model = load_karpathy_model("stories42M.bin", "tokenizer.bin");

julia> sample(model, "Julia is the best"; temperature = 0.5f0)
<s>
Julia is the best. She has to do what I say. She is not happy.
[...]
```

Here is a demo of a Llama2 7B model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML:

**Only the q4_K_S variant is supported.**

```
julia> model = load_ggml_model("llama-2-7b-chat.ggmlv3.q4_K_S.bin");

julia> sample(model, "The Julia programming language is")
<s>
The Julia programming language is a high-level, high-performance dynamic programming language for technical computing.

```
Thanks to weight quantization, this only requires a machine with 8GB RAM.
