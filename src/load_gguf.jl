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

function TransformerLayerWeights(tensor_dict::Dict{String,Any}, layer_index::Int)
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

function TransformerWeights(tensor_dict::Dict{String,Any}, layer_count::Int)
    layers = [TransformerLayerWeights(tensor_dict, 1)]

    for i in 2:layer_count
        push!(layers, TransformerLayerWeights(tensor_dict, i))
    end

    return TransformerWeights(;
        token_embedding_table = tensor_dict["token_embd.weight"],
        rms_final_weight      = tensor_dict["output_norm.weight"],
        output_weight         = tensor_dict["output.weight"],
        layers,
    )
end

function align_offset(offset, alignment)
    return offset + (alignment - (offset % alignment)) % alignment
end

function read_ggml_tensor(tensor_type::GGML_TYPE, size, file::IOStream)
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

function load_gguf_model(filename::AbstractString)
    time = time_ns()
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
            tensor_dict[tensor_info.name] = read_ggml_tensor(tensor_info.typ, tensor_info.dimensions, file)
        end
    end

    metadata_kv = header.metadata_kv

    # read tokenizer
    id_to_token = metadata_kv["tokenizer.ggml.tokens"]
    token_types = metadata_kv["tokenizer.ggml.token_type"]
    
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

    token_to_id = Dict{String,Int}(token => id for (id, token) in enumerate(id_to_token))
    token_scores = metadata_kv["tokenizer.ggml.scores"]
    # TODO: also store tokenizer.ggml.bos_token_id, tokenizer.ggml.eos_token_id in tokenizer
    tokenizer = BPETokenizer(id_to_token, token_to_id, token_scores)

    config = ModelConfig(;
        dim         = metadata_kv["llama.embedding_length"],
        hidden_dim  = metadata_kv["llama.feed_forward_length"],
        n_layers    = metadata_kv["llama.block_count"],
        n_heads     = metadata_kv["llama.attention.head_count"],
        n_kv_heads  = metadata_kv["llama.attention.head_count_kv"],
        vocab_size  = length(id_to_token),
        seq_len     = 512, # metadata_kv["llama.context_length"],
    )

    weights = TransformerWeights(tensor_dict, Int(header.metadata_kv["llama.block_count"]))

    @info "Loaded model in $((time_ns() - time) / 1e9) seconds"
    return LanguageModel(config, tokenizer, weights)
end
