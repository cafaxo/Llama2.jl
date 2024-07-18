@enum GGUF_METADATA_VALUE_TYPE::UInt32 begin
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0
    GGUF_METADATA_VALUE_TYPE_INT8 = 1
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2
    GGUF_METADATA_VALUE_TYPE_INT16 = 3
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4
    GGUF_METADATA_VALUE_TYPE_INT32 = 5
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6

    # 1-byte value where 0 is false and 1 is true.
    # Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7

    # The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8

    # The value is an array of other values, with the length and type prepended.
    # Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9

    GGUF_METADATA_VALUE_TYPE_UINT64 = 10
    GGUF_METADATA_VALUE_TYPE_INT64 = 11
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12
end

@enum GGML_TYPE::UInt32 begin
    GGML_TYPE_F32  = 0
    GGML_TYPE_F16  = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    # k-quantizations
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
end

@enum LLAMA_TOKEN_TYPE begin
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0
    LLAMA_TOKEN_TYPE_NORMAL       = 1
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2
    LLAMA_TOKEN_TYPE_CONTROL      = 3
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4
    LLAMA_TOKEN_TYPE_UNUSED       = 5
    LLAMA_TOKEN_TYPE_BYTE         = 6
end

function read_gguf_string(file)
    len = read(file, UInt64)
    return String(read(file, len))
end

function read_gguf_metadata_value(file, value_type)
    if value_type == GGUF_METADATA_VALUE_TYPE_STRING
        return read_gguf_string(file)
    elseif value_type == GGUF_METADATA_VALUE_TYPE_UINT32
        return read(file, UInt32)
    elseif value_type == GGUF_METADATA_VALUE_TYPE_INT32
        return read(file, Int32)
    elseif value_type == GGUF_METADATA_VALUE_TYPE_FLOAT32
        return read(file, Float32)
    elseif value_type == GGUF_METADATA_VALUE_TYPE_ARRAY
        array_value_type = GGUF_METADATA_VALUE_TYPE(read(file, UInt32))
        array_len = read(file, UInt64)
        return [read_gguf_metadata_value(file, array_value_type) for _ in 1:array_len]
    elseif value_type == GGUF_METADATA_VALUE_TYPE_BOOL
        return read(file, Bool)
    end

    error("reading metadata value of type $value_type is not implemented yet")
end

struct GGUFHeader
    tensor_count::UInt64
    metadata_kv::Dict{String,Any}
end

function read_gguf_header(file)
    magic = String(read(file, 4))

    if magic != "GGUF"
        error("Not a GGUF file")
    end

    version = read(file, UInt32)
    tensor_count = read(file, UInt64)
    metadata_kv_count = read(file, UInt64)
    metadata_kv = Dict{String,Any}()

    for _ in 1:metadata_kv_count
        # read key string
        key = read_gguf_string(file)
        value_type = GGUF_METADATA_VALUE_TYPE(read(file, UInt32))
        value = read_gguf_metadata_value(file, value_type)
        metadata_kv[key] = value
    end

    return GGUFHeader(tensor_count, metadata_kv)
end

@kwdef struct GGUFTensorInfo
    name::String
    dimensions::Vector{UInt64}
    typ::GGML_TYPE

    # This offset is relative to `tensor_data`, not to the start
    # of the file, to make it easier for writers to write the file.
    # Readers should consider exposing this offset relative to the
    # file to make it easier to read the data.
    #
    # Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    offset::UInt64
end

function read_gguf_tensor_info(file)
    name = read_gguf_string(file)
    n_dimensions = read(file, UInt32)
    dimensions = [read(file, UInt64) for _ in 1:n_dimensions]
    typ = GGML_TYPE(read(file, UInt32))
    offset = read(file, UInt64)

    return GGUFTensorInfo(; name, dimensions, typ, offset)
end

function align_offset(offset, alignment)
    return offset + (alignment - (offset % alignment)) % alignment
end

function _read_ggml_tensor(tensor_type::GGML_TYPE, size, file::IOStream)
    N = length(size)

    if tensor_type == GGML_TYPE_F32
        tensor = Array{Float32,N}(undef, size...)
    elseif tensor_type == GGML_TYPE_Q4_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = Array{block_q4_K,N}(undef, size...)
    elseif tensor_type == GGML_TYPE_Q5_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = Array{block_q5_K,N}(undef, size...)
    elseif tensor_type == GGML_TYPE_Q6_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = Array{block_q6_K,N}(undef, size...)
    else
        error("tensor type $tensor_type not implemented")
    end

    read!(file, tensor)
    return tensor
end

function _read_ggml_tensor_mmap(tensor_type::GGML_TYPE, size, file::IOStream)
    N = length(size)

    if tensor_type == GGML_TYPE_F32
        size = Tuple(size)
        tensor = mmap(file, Array{Float32,N}, size)
    elseif tensor_type == GGML_TYPE_Q4_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = mmap(file, Array{block_q4_K,N}, size)
    elseif tensor_type == GGML_TYPE_Q5_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = mmap(file, Array{block_q5_K,N}, size)
    elseif tensor_type == GGML_TYPE_Q6_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = mmap(file, Array{block_q6_K,N}, size)
    else
        error("tensor type $tensor_type not implemented")
    end

    seek(file, position(file) + sizeof(tensor))

    return tensor
end

function read_ggml_tensor(tensor_type::GGML_TYPE, size, file::IOStream, mmap)
    if mmap
        return _read_ggml_tensor_mmap(tensor_type, size, file)
    end

    return _read_ggml_tensor(tensor_type, size, file)
end

# this undoes whatever https://github.com/openai/gpt-2/blob/master/src/encoder.py does
function gpt2_decoder()
    bs = collect(Int('!'):Int('~')) ∪ collect(Int('¡'):Int('¬')) ∪ collect(Int('®'):Int('ÿ'))
    cs = copy(bs)
    n = 0
    for b in 0:255
        if b ∉ bs
            push!(bs, b)
            push!(cs, 2^8 + n)
            n += 1
        end
    end
    
    return Dict(Char.(cs) .=> Char.(bs))
end

function build_llama_weights_layer(tensor_dict::Dict{String,Any}, layer_index::Int)
    if !haskey(tensor_dict, "blk.$(layer_index-1).attn_q.weight")
        error("missing blk.$(layer_index-1) weights")
    end

    return TransformerLayerWeights(;
        rms_att_weight = tensor_dict["blk.$(layer_index-1).attn_norm.weight"],
        rms_ffn_weight = tensor_dict["blk.$(layer_index-1).ffn_norm.weight"],
        wq             = tensor_dict["blk.$(layer_index-1).attn_q.weight"],
        wk             = tensor_dict["blk.$(layer_index-1).attn_k.weight"],
        wv             = tensor_dict["blk.$(layer_index-1).attn_v.weight"],
        wo             = tensor_dict["blk.$(layer_index-1).attn_output.weight"],
        w1             = tensor_dict["blk.$(layer_index-1).ffn_gate.weight"],
        w2             = tensor_dict["blk.$(layer_index-1).ffn_down.weight"],
        w3             = tensor_dict["blk.$(layer_index-1).ffn_up.weight"],
    )
end

function build_llama(metadata_kv, tensor_dict, vocab_size)
    config = ModelConfig(;
        dim            = metadata_kv["llama.embedding_length"],
        hidden_dim     = metadata_kv["llama.feed_forward_length"],
        n_layers       = metadata_kv["llama.block_count"],
        n_heads        = metadata_kv["llama.attention.head_count"],
        n_kv_heads     = metadata_kv["llama.attention.head_count_kv"],
        vocab_size,
        seq_len        = 512, # metadata_kv["llama.context_length"],
        rope_freq_base = get(metadata_kv, "llama.rope.freq_base", 10000.0f0),
        rope_is_neox   = false,
    )

    layer_count = Int(metadata_kv["llama.block_count"])
    layers = TransformerLayerWeights[build_llama_weights_layer(tensor_dict, i) for i in 1:layer_count]

    weights = TransformerWeights(;
        token_embedding_table = tensor_dict["token_embd.weight"],
        rms_final_weight      = tensor_dict["output_norm.weight"],
        output_weight         = tensor_dict["output.weight"],
        layers,
    )

    return config, weights
end

function build_phi3_weights_layer(tensor_dict::Dict{String,Any}, layer_index::Int)
    if !haskey(tensor_dict, "blk.$(layer_index-1).attn_qkv.weight")
        error("missing blk.$(layer_index-1) weights")
    end

    wqkv = tensor_dict["blk.$(layer_index-1).attn_qkv.weight"]
    n = size(wqkv, 2) ÷ 3
    @assert n == 3072

    wq = view(wqkv, :, 0*n+1:1*n)
    wk = view(wqkv, :, 1*n+1:2*n)
    wv = view(wqkv, :, 2*n+1:3*n)

    # these are combined in phi3 for some reason
    gate_and_up = tensor_dict["blk.$(layer_index-1).ffn_up.weight"]
    n = size(gate_and_up, 2) ÷ 2

    # ffn_gate
    w1 = view(gate_and_up, :, 0*n+1:1*n)

    # ffn_up
    w3 = view(gate_and_up, :, 1*n+1:2*n)

    return TransformerLayerWeights(;
        rms_att_weight = tensor_dict["blk.$(layer_index-1).attn_norm.weight"],
        rms_ffn_weight = tensor_dict["blk.$(layer_index-1).ffn_norm.weight"],
        wq,
        wk,
        wv,
        wo             = tensor_dict["blk.$(layer_index-1).attn_output.weight"],
        w1,
        w2             = tensor_dict["blk.$(layer_index-1).ffn_down.weight"],
        w3,
    )
end

function build_phi3(metadata_kv, tensor_dict, vocab_size)
    config = ModelConfig(;
        dim            = metadata_kv["phi3.embedding_length"],
        hidden_dim     = metadata_kv["phi3.feed_forward_length"],
        n_layers       = metadata_kv["phi3.block_count"],
        n_heads        = metadata_kv["phi3.attention.head_count"],
        n_kv_heads     = metadata_kv["phi3.attention.head_count_kv"],
        vocab_size,
        seq_len        = 512, # metadata_kv["phi3.context_length"],
        rope_freq_base = get(metadata_kv, "phi3.rope.freq_base", 10000.0f0),
        rope_is_neox   = true,
    )

    layer_count = Int(metadata_kv["phi3.block_count"])
    layers = TransformerLayerWeights[build_phi3_weights_layer(tensor_dict, i) for i in 1:layer_count]

    weights = TransformerWeights(;
        token_embedding_table = tensor_dict["token_embd.weight"],
        rms_final_weight      = tensor_dict["output_norm.weight"],
        output_weight         = tensor_dict["output.weight"],
        layers,
    )

    return config, weights
end

function load_gguf_model(filename::AbstractString; mmap=true)
    header = nothing
    tensor_dict = nothing

    open(filename) do file
        header = read_gguf_header(file)

        tensor_info_list = [read_gguf_tensor_info(file) for _ in 1:header.tensor_count]
        tensor_dict = Dict{String,Any}()

        alignment = get(header.metadata_kv, "general.alignment", UInt32(32))
        pad_offset = align_offset(position(file), alignment)

        # read tensors
        @showprogress desc="Loading model..." for tensor_info in tensor_info_list
            seek(file, pad_offset + tensor_info.offset)
            tensor_dict[tensor_info.name] = read_ggml_tensor(tensor_info.typ, tensor_info.dimensions, file, mmap)
        end
    end

    metadata_kv = header.metadata_kv

    # read tokenizer
    id_to_token = metadata_kv["tokenizer.ggml.tokens"]
    token_types = metadata_kv["tokenizer.ggml.token_type"]

    tokenizer_model = metadata_kv["tokenizer.ggml.model"]
    
    if tokenizer_model == "llama"
        for i in 1:length(id_to_token)
            token_type = LLAMA_TOKEN_TYPE(token_types[i])

            # fix whitespace in normal tokens TODO: figure out why llama.cpp does this
            if token_type == LLAMA_TOKEN_TYPE_NORMAL
                id_to_token[i] = replace(id_to_token[i], "\xe2\x96\x81" => " ")
            end

            # fix byte tokens
            if token_type == LLAMA_TOKEN_TYPE_BYTE
                id_to_token[i] = String([parse(UInt8, id_to_token[i][2:end-1])])
            end
        end
    elseif tokenizer_model == "gpt2"
        gpt2_decode_dict = gpt2_decoder()

        for i in 1:length(id_to_token)
            token_type = LLAMA_TOKEN_TYPE(token_types[i])

            if token_type == LLAMA_TOKEN_TYPE_NORMAL
                id_to_token[i] = String([gpt2_decode_dict[c] for c in id_to_token[i]])
            end
        end
    else
        error("unsupported tokenizer model: $(tokenizer_model)")
    end

    token_to_id = Dict{String,Int}(token => id for (id, token) in enumerate(id_to_token))
    tokenizer = BPETokenizer(
        id_to_token,
        token_to_id,
        metadata_kv["tokenizer.ggml.scores"],
        metadata_kv["tokenizer.ggml.bos_token_id"] + 1, # convert to one-based indexing
        metadata_kv["tokenizer.ggml.eos_token_id"] + 1,
    )

    vocab_size = length(id_to_token)
    model_architecture = metadata_kv["general.architecture"]

    if model_architecture == "llama"
        config, weights = build_llama(metadata_kv, tensor_dict, vocab_size)
    elseif model_architecture == "phi3"
        config, weights = build_phi3(metadata_kv, tensor_dict, vocab_size)
    else
        error("unsupported model architecture: $(model_architecture)")
    end

    return LanguageModel(config, tokenizer, weights)
end
