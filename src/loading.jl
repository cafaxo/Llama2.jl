function read_config(f::IOStream)
    Config(read(f, Int32),
        read(f, Int32),
        read(f, Int32),
        read(f, Int32),
        read(f, Int32),
        read(f, Int32),
        read(f, Int32))
end

function checkpoint_init_weights!(w::TransformerWeights, f::IOStream)
    read!(f, w.token_embedding_table)
    read!(f, w.rms_att_weight)
    read!(f, w.wq)
    read!(f, w.wk)
    read!(f, w.wv)
    read!(f, w.wo)
    read!(f, w.rms_ffn_weight)
    read!(f, w.w1)
    read!(f, w.w2)
    read!(f, w.w3)
    read!(f, w.rms_final_weight)
    read!(f, w.freq_cis_real)
    read!(f, w.freq_cis_imag)
    return nothing
end

"""
    load_model(checkpoint_filename::AbstractString)

Loads a model from a file. The file should be a `model.bin` file
"""
function load_model(checkpoint_filename::AbstractString)
    config, weights = nothing, nothing
    open(checkpoint_filename) do file
        config = read_config(file)
        weights = TransformerWeights(config)
        checkpoint_init_weights!(weights, file)
    end

    return config, weights
end

"""
    load_tokenizer(tokenizer_filename::AbstractString, vocab_size::Int)

Loads a tokenizer from a file. The file should be a `tokenizer.bin` file
"""
function load_tokenizer(tokenizer_filename::AbstractString, vocab_size::Int)
    vocab = Vector{Vector{UInt8}}(undef, vocab_size)
    vocab_scores = Vector{Float32}(undef, vocab_size)
    max_token_length = 1

    open(tokenizer_filename) do file
        max_token_length = read(file, Int32)
        for i in 1:(vocab_size)
            vocab_scores[i] = read(file, Float32)
            len = read(file, Int32)
            vocab[i] = read(file, len)
        end
    end

    return TokenizerConfig(vocab, vocab_scores, max_token_length)
end

"""
    load_model(checkpoint_filename::AbstractString, tokenizer_filename::AbstractString)

Loads a pretrained model and tokenizer from the corresponding files
"""
function load_model(checkpoint_filename::AbstractString, tokenizer_filename::AbstractString)
    @assert isfile(checkpoint_filename) "`checkpoint_filename` must be an existing file (eg, `model.bin`)"
    @assert isfile(tokenizer_filename) "`tokenizer_filename` must be an existing file (eg, `tokenizer.bin`)"

    config, weights = load_model(checkpoint_filename)
    tokenizer = load_tokenizer(tokenizer_filename, config.vocab_size)
    return TrainedModel(config, weights, tokenizer)
end
