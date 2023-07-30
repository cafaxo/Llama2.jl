# Llama2.jl

The `src/inference.jl` file is currently a close port of Andrej Karpathy's llama.c (https://github.com/karpathy/llama2.c).

## Goals of this project

The goal is to provide code that is *readable* and *self-contained* (uses minimal dependencies).

## Installation

This package is not yet registered. It can be installed by running
```
pkg> add https://github.com/cafaxo/Llama2.jl
```

## Usage

Download one of the llama2.c models from https://huggingface.co/karpathy/tinyllamas and the [tokenizer.bin](https://github.com/karpathy/llama2.c/raw/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin) file.

The model can be sampled with
```
julia> using Llama2

julia> sample("stories42M.bin", "tokenizer.bin")
```

Alternatively, you can load the model and tokenizer into memory and sample repeatedly from it with different prompts:   
```
julia> using Llama2

julia> model = load_model("stories42M.bin", "tokenizer.bin")

julia> sample(model, "Julia is the best"; temperature = 0.5f0)
```