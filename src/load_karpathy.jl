read_karpathy_config(f::IOStream) = ModelConfig(
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    read(f, Int32),
    10000.0f0,
)

TransformerLayerWeights(p::ModelConfig) = TransformerLayerWeights(;
    rms_att_weight = zeros(Float32, p.dim),
    rms_ffn_weight = zeros(Float32, p.dim),
    wq             = zeros(Float32, p.dim, p.dim),
    wk             = zeros(Float32, p.dim, p.dim),
    wv             = zeros(Float32, p.dim, p.dim),
    wo             = zeros(Float32, p.dim, p.dim),
    w1             = zeros(Float32, p.dim, p.hidden_dim),
    w2             = zeros(Float32, p.hidden_dim, p.dim),
    w3             = zeros(Float32, p.dim, p.hidden_dim),
)

TransformerWeights(p::ModelConfig) = TransformerWeights(;
    token_embedding_table = zeros(Float32, p.dim, p.vocab_size),
    layers                = [TransformerLayerWeights(p) for _ in 1:p.n_layers],
    rms_final_weight      = zeros(Float32, p.dim),
    output_weight         = zeros(Float32, p.dim, p.vocab_size),
)

function read_karpathy_weights(f::IOStream, config::ModelConfig)
    n_layers = config.n_layers
    w = TransformerWeights(config)

    read!(f, w.token_embedding_table)

    read_into_layer! = gf -> begin
        for l in 1:n_layers
            read!(f, gf(w.layers[l]))
        end
    end

    read_into_layer!(l -> l.rms_att_weight)
    read_into_layer!(l -> l.wq)
    read_into_layer!(l -> l.wk)
    read_into_layer!(l -> l.wv)
    read_into_layer!(l -> l.wo)
    read_into_layer!(l -> l.rms_ffn_weight)
    read_into_layer!(l -> l.w1)
    read_into_layer!(l -> l.w2)
    read_into_layer!(l -> l.w3)

    read!(f, w.rms_final_weight)

    copyto!(w.output_weight, w.token_embedding_table)

    return w
end

function load_karpathy_tokenizer(filename::AbstractString, vocab_size::Int)
    id_to_token = Vector{String}(undef, vocab_size)
    token_to_id = Dict{String,Int}()
    token_scores = Vector{Float32}(undef, vocab_size)

    open(filename) do file
        max_token_length = read(file, Int32)
        for i in 1:vocab_size
            token_scores[i] = read(file, Float32)
            len = read(file, Int32)
            word = String(read(file, len))
            id_to_token[i] = word
            token_to_id[word] = i
        end
    end

    return BPETokenizer(id_to_token, token_to_id, token_scores, 2, 3)
end

function load_karpathy_model(
        checkpoint_filename::AbstractString,
        tokenizer_filename::AbstractString,
    )

    config = nothing
    weights = nothing

    # read in the model.bin file
    open(checkpoint_filename) do file
        config = read_karpathy_config(file)
        weights = read_karpathy_weights(file, config)
    end

    # read in the tokenizer.bin file
    tokenizer = load_karpathy_tokenizer(tokenizer_filename, config.vocab_size)

    return LanguageModel(config, tokenizer, weights)
end
