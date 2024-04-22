module Llama2

using LinearAlgebra
using StatsBase
using Printf
using ProgressMeter
using LoopVectorization
using Random
using Distributions
using Mmap

export ModelConfig, CharTokenizer, LanguageModel
export load_gguf_model, load_karpathy_model, encode, sample
export train

# quantization
include("quantization/utils.jl")
include("quantization/common.jl")
include("quantization/q4.jl")
include("quantization/q6.jl")
include("quantization/q8.jl")
include("quantization/vecdot.jl")

# inference
include("tokenizer.jl")
include("matmul.jl")
include("inference.jl")

# model loading
include("load_gguf.jl")
include("load_karpathy.jl")

# training
include("training/graph.jl")
include("training/tensor.jl")
include("training/optimizers.jl")
include("training/ops/add.jl")
include("training/ops/attention.jl")
include("training/ops/dense.jl")
include("training/ops/kl_divergence.jl")
include("training/ops/mul.jl")
include("training/ops/pointwise.jl")
include("training/ops/rmsnorm.jl")
include("training/ops/rope.jl")
include("training/ops/softmax.jl")
include("training/model.jl")

end # module Llama2
