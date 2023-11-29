
const GGUF_MAGIC             = 0x46554747
const GGUF_VERSION           = 3  # 2,3
const GGUF_DEFAULT_ALIGNMENT = 32

@enum GGML_TYPE begin
    GGML_TYPE_F32  = 0
    GGML_TYPE_F16  = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    # GGML_TYPE_Q4_2 = 4, support has been removed
    # GGML_TYPE_Q4_3 (5) support has been removed
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
    GGML_TYPE_I8
    GGML_TYPE_I16
    GGML_TYPE_I32
    GGML_TYPE_COUNT
end

@enum GGUF_METADATA_VALUE_TYPE begin
    # The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0
    # The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1
    # The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2
    # The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3
    # The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4
    # The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5
    # The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
    # The value is a boolean.
    # 1-byte value where 0 is false and 1 is true.
    # Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7
    # The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8
    # The value is an array of other values, with the length and type prepended.
    # Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9
    # The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10
    # The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11
    # The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12
end

@enum FILE_TYPE begin
    # Q:quant, K:K-quant, S:single, M:multi/mixture
    ALL_F32 = 0
    MOSTLY_F16 = 1
    MOSTLY_Q4_0 = 2
    MOSTLY_Q4_1 = 3
    MOSTLY_Q4_1_SOME_F16 = 4
    # MOSTLY_Q4_2 = 5 (support removed)
    # MOSTLY_Q4_3 = 6 (support removed)
    MOSTLY_Q8_0 = 7
    MOSTLY_Q5_0 = 8
    MOSTLY_Q5_1 = 9
    MOSTLY_Q2_K = 10
    MOSTLY_Q3_K_S = 11
    MOSTLY_Q3_K_M = 12  # Q3_K
    MOSTLY_Q3_K_L = 13
    MOSTLY_Q4_K_S = 14
    MOSTLY_Q4_K_M = 15  # Q4_K
    MOSTLY_Q5_K_S = 16
    MOSTLY_Q5_K_M = 17  # Q5_K
    MOSTLY_Q6_K = 18
end

@enum TOKEN_TYPE begin
    normal=1 
    unknown=2
    control=3 
    user_defined=4
    unused=5
    byte=6
end

mutable struct GGUF_STRING
    len::UInt64
    string::String
end

mutable struct GGUFMetadataValue
    type::GGUF_METADATA_VALUE_TYPE
    len::UInt64
    value::Union{UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, UInt64, Int64, Float64, Bool, String, Vector{GGUFMetadataValue}}
end

mutable struct GGUFMetadataValueKV
    key::GGUF_STRING
    value_type::GGUF_METADATA_VALUE_TYPE
    value::GGUFMetadataValue
end

mutable struct GGUFHeader
    magic::UInt32
    version::UInt32
    tensor_count::UInt64
    metadata_kv_count::UInt64
    metadata_kv::Array{GGUFMetadataValueKV, 1}
end

mutable struct GGUFTensorInfo
    name::GGUF_STRING
    n_dimensions::UInt32
    dimensions::Array{UInt64, 1}
    type::GGML_TYPE
    offset::UInt64
end

mutable struct GGUFFile
    header::GGUFHeader
    tensor_infos::Array{GGUFTensorInfo, 1}
    tensor_data::Array{UInt8, 1}
end

function read_gguf_header(file::IOStream)
    header = GGUFHeader(
        read(file, UInt32),
        read(file, UInt32),
        read(file, UInt64),
        read(file, UInt64),
        []
    )
    for i in 1:header.metadata_kv_count   # GGUFMetadataValueKV  19个
        key_len = read(file, UInt64)
        key_str = String(read(file, key_len))
        key = GGUF_STRING(key_len, key_str)

        value_type = GGUF_METADATA_VALUE_TYPE(read(file, UInt32))
        if value_type == GGUF_METADATA_VALUE_TYPE_STRING  # 8
            value_len = read(file, UInt64)
            value = String(read(file, value_len))
        elseif value_type == GGUF_METADATA_VALUE_TYPE_UINT32  # 4
            value = read(file, UInt32)
            value_len = 4
        elseif value_type == GGUF_METADATA_VALUE_TYPE_UINT64   # 10
            value = read(file, UInt64)
            value_len = 8
        elseif value_type == GGUF_METADATA_VALUE_TYPE_FLOAT32   # 6
            value = read(file, Float32)
            value_len = 4
        elseif value_type == GGUF_METADATA_VALUE_TYPE_ARRAY   # 9
            array_value_type = GGUF_METADATA_VALUE_TYPE(read(file, UInt32))
            array_len = read(file, UInt64)  # array len 
            # println(array_len, ", ", array_value_type)
            array = Vector{GGUFMetadataValue}(undef, array_len)
            for j in 1:array_len
                if array_value_type == GGUF_METADATA_VALUE_TYPE_STRING  # 8
                    value_len = read(file, UInt64)
                    value = String(read(file, value_len))
                elseif array_value_type == GGUF_METADATA_VALUE_TYPE_UINT32  # 4
                    value = read(file, UInt32)
                    value_len = 4
                elseif array_value_type == GGUF_METADATA_VALUE_TYPE_INT32  # 4
                    value = read(file, Int32)
                    value_len = 4
                elseif array_value_type == GGUF_METADATA_VALUE_TYPE_FLOAT32   # 10
                    value = read(file, Float32)
                    value_len = 4
                else
                    println(array_value_type, "===========================")
                    value = ""   
                    value_len = 0
                end
                array_value = GGUFMetadataValue(array_value_type, value_len, value)
                array[j] = array_value
            end
            value = array
        else
            value = ""

        end

        # println(i, ", ", key.string, ", ", value_type, ",", value_len, "================")
        if key.string == "general.name"
            println("general.name:", value)
        elseif key.string == "general.architecture"
            println("general.architecture:", value)
        elseif key.string == "general.quantization_version"
            println("quantization_version:", value)
        elseif key.string == "general.file_type"
            println("file_type:", FILE_TYPE(value))
        elseif key.string == "tokenizer.ggml.model"
            println("tokenizer.ggml.model:", value)
        elseif key.string == "tokenizer.ggml.token_type"
            println("tokenizer.ggml.token_type:", length(value), ",", value[1].value, ",", value[2].value, ",", value[4].value, ",", value[6].value, ",", value[8].value)
        elseif key.string == "tokenizer.ggml.tokens"
            println("tokenizer.ggml.tokens:", length(value), ",", value[1].value, value[2].value, ",", value[4].value, ",", value[6].value, ",", value[8].value)
        elseif key.string == "tokenizer.ggml.bos_token_id"
            println("tokenizer.ggml.bos_token_id:", value)
        elseif key.string == "tokenizer.ggml.eos_token_id"
            println("tokenizer.ggml.eos_token_id:", value)
        end

        gguf_value = GGUFMetadataValue(value_type, value_len, value)

        kv = GGUFMetadataValueKV(key, value_type, gguf_value)
        push!(header.metadata_kv, kv)
    end

    return header
end

function read_gguf_tokenizer(header)
    tokens, scores, token_types = nothing, nothing, nothing
    for kv in header.metadata_kv
        if kv.key.string == "tokenizer.ggml.model"
            model = kv.value.value
        elseif kv.key.string == "tokenizer.ggml.token_type"
            token_types = kv.value.value
        elseif kv.key.string == "tokenizer.ggml.tokens"
            tokens = kv.value.value
        elseif kv.key.string == "tokenizer.ggml.scores"
            scores = kv.value.value
        end
    end
    n_vocab = length(tokens)   # tokens number

    id_to_token = Vector{String}(undef, n_vocab)   # String
    token_to_id = Dict{String,Int}()
    token_scores = Vector{Float32}(undef, n_vocab)
    
    for i in 1:n_vocab
        word = tokens[i].value
        score = scores[i].value
        type = token_types[i].value   # type: 1,2,3,6

        word = replace(word, "▁"=>" ")
        # println(i,  ", ", word, ", score:", score, ", type:", type,"===============")
        # if i in [36, 1,2,3,4,5,32,36]   # i in [36, 2, 3,4,5]  word == "▁"
        #     println(i,  ", ", word, ", score:", score, ", type:", type,"===============")  # type: 1,3,6
        # end
        id_to_token[i] = word
        token_scores[i] = score
        token_to_id[word] = i
    end

    return BPETokenizer(id_to_token, token_to_id, token_scores)
end

function read_gguf_tensor(tensor_type::GGML_TYPE, size::Tuple, file::IOStream)
    N = length(size)

    if tensor_type == GGML_TYPE_F32
        tensor = Array{Float32,N}(undef, size)
    elseif tensor_type == GGML_TYPE_Q4_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = Array{block_q4_K,N}(undef, size)
    elseif tensor_type == GGML_TYPE_Q5_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = Array{block_q5_K,N}(undef, size)
    elseif tensor_type == GGML_TYPE_Q6_K
        @assert size[1] % QK_K == 0
        size = (size[1] ÷ QK_K, size[2:end]...)
        tensor = Array{block_q6_K,N}(undef, size)
    else
        error("tensor type $tensor_type not implemented")
    end

    read!(file, tensor)
    # println(size, ", ", tensor_type, ", ", length(tensor))
    return tensor
end

function read_gguf_tensor_dict(file::IOStream; tensor_infos=[], show_progress=true)
    if show_progress
        pos = position(file)
        seekend(file)
        file_size = position(file)
        seek(file, pos)

        progress = Progress(file_size, "Loading model...")
        update!(progress, position(file))
    end

    tensor_dict = Dict{String,Any}()
    tensor_infos = sort(tensor_infos, by=x->x.offset)
    for tensor_info in tensor_infos
        if tensor_info.offset == 0
            pos = position(file)
            seek(file, pos + (-pos & 31))   # skip to the next multiple of 32 bytes.  GGUF_DEFAULT_ALIGNMENT
        else
            seek(file, tensor_info.offset)
        end
        dimensions = Tuple(tensor_info.dimensions)
        tensor_dict[tensor_info.name.string] = read_gguf_tensor(tensor_info.type, dimensions, file)
        show_progress && update!(progress, position(file))
    end

    return tensor_dict
end

function TransformerLayerWeights_1(ggml_dict::Dict{String,Any}, layer_index::Int)
    if !haskey(ggml_dict, "blk.$(layer_index-1).attn_q.weight")
        error("missing blk.$(layer_index-1) weights")
    end

    layer_weight = TransformerLayerWeights_gguf{Llama2.block_q4_K, Llama2.block_q5_K}(;
        rms_att_weight = ggml_dict["blk.$(layer_index-1).attn_norm.weight"],
        rms_ffn_weight = ggml_dict["blk.$(layer_index-1).ffn_norm.weight"],
        wq             = ggml_dict["blk.$(layer_index-1).attn_q.weight"],
        wk             = ggml_dict["blk.$(layer_index-1).attn_k.weight"],
        wv             = ggml_dict["blk.$(layer_index-1).attn_v.weight"],
        wo             = ggml_dict["blk.$(layer_index-1).attn_output.weight"],
        w1             = ggml_dict["blk.$(layer_index-1).ffn_up.weight"],
        w2             = ggml_dict["blk.$(layer_index-1).ffn_down.weight"],
        w3             = ggml_dict["blk.$(layer_index-1).ffn_gate.weight"],
        )
    return layer_weight
end

function TransformerWeights_1(ggml_dict::Dict{String,Any}, layer_count::Int)
    # println(keys(ggml_dict), layer_count)  # 32
    # layers = [TransformerLayerWeights_1(ggml_dict, 1)]
    layers = Vector{TransformerLayerWeights_gguf}(undef, layer_count)

    for i in 1:layer_count
        layers[i] = TransformerLayerWeights_1(ggml_dict, i)
    end

    return TransformerWeights(;
        token_embedding_table = ggml_dict["token_embd.weight"],
        rms_final_weight      = ggml_dict["output_norm.weight"],
        output_weight         = ggml_dict["output.weight"],
        layers                = layers,
    )
end

function load_gguf_model(filename::AbstractString)
    header = nothing
    tokenizer = nothing
    tensor_infos = Vector{GGUFTensorInfo}()
    tensor_dict = nothing

    open(filename) do file
        magic = read(file, UInt32)

        if magic != GGUF_MAGIC
            error("Only the GGUF file format is supported,  magic")
        end

        version = read(file, UInt32)

        if version ∉ [2,3]   # != GGUF_VERSION
            error("Only the GGUF file format is supported,  version")
        end
        seek(file, 0)
        header = read_gguf_header(file)

        tokenizer = read_gguf_tokenizer(header)
    
        # read tensor_infos
        for i in 1:header.tensor_count
            key_len = read(file, UInt64)
            key_str = String(read(file, key_len))
            name = GGUF_STRING(key_len, key_str)
            n_dimensions = read(file, UInt32)
            tensor_info = GGUFTensorInfo(
                name,
                n_dimensions,
                [read(file, UInt64) for _ in 1:n_dimensions],
                GGML_TYPE(read(file, UInt32)),
                read(file, UInt64)
            )
            # println(i,", ",  tensor_info.name.string, ", ", tensor_info.n_dimensions, ", ", tensor_info.type, ", ", tensor_info.dimensions) 
            push!(tensor_infos, tensor_info)
        end
       
        # read tensor_data
        tensor_dict = read_gguf_tensor_dict(file; tensor_infos)

    end
    # gguf_file = GGUFFile(header, tensor_infos, tensor_data)

    # compute hidden dim
    context_length = header.metadata_kv[3].value.value
    embedding_length = header.metadata_kv[4].value.value
    block_count = header.metadata_kv[5].value.value   # n_layers
    feed_forward_length = header.metadata_kv[6].value.value
    dimension_count = header.metadata_kv[7].value.value
    head_count = header.metadata_kv[8].value.value
    head_count_kv = header.metadata_kv[9].value.value
    n_vocab = length(header.metadata_kv[14].value.value)
    # println(n_vocab)
    weights = TransformerWeights_1(tensor_dict, Int(block_count))

    config = ModelConfig(;
        dim         = embedding_length,
        hidden_dim  = feed_forward_length,
        n_layers    = block_count,
        n_heads     = head_count,
        n_kv_heads  = head_count_kv,
        vocab_size  = n_vocab,
        seq_len     = 512,  # FIXME: not sure what to put here.  context_length
    )

    return LanguageModel(config, tokenizer, weights)

end

# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
