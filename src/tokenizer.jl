abstract type Tokenizer end

struct BPETokenizer <: Tokenizer
    id_to_token::Vector{String}
    token_to_id::Dict{String,Int}
    token_scores::Vector{Float32}
    bos_token_id::Int
    eos_token_id::Int
end

function encode(text::AbstractString, tokenizer::BPETokenizer; pad_input=true)
    (; id_to_token, token_to_id, token_scores) = tokenizer

    tokens = Int[]

    if isempty(text)
        return tokens
    end

    if pad_input
        push!(tokens, token_to_id[" "])
    end

    # encode every individual codeunit in the input string
    for codeunit in codeunits(text)
        push!(tokens, token_to_id[String(UInt8[codeunit])])
    end

    while true
        best_score = -Inf32
        best_id = -1
        best_idx = -1

        for i in 1:(length(tokens) - 1)
            # check if we can merge the pair (tokens[i], tokens[i+1])
            id = get(token_to_id, id_to_token[tokens[i]] * id_to_token[tokens[i+1]], nothing)

            if !isnothing(id) && token_scores[id] > best_score
                best_score = token_scores[id]
                best_id = id
                best_idx = i
            end
        end

        if best_idx == -1
            break # we couldn't find any more pairs to merge, so we're done
        end

        # merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id

        # delete token at position best_idx+1, shift the entire sequence back 1
        deleteat!(tokens, best_idx+1)
    end

    return tokens
end

struct CharTokenizer <: Tokenizer
    id_to_token::Vector{String}
    token_to_id::Dict{String,Int}
    bos_token_id::Int
    eos_token_id::Int
end

function CharTokenizer(text::AbstractString)
    id_to_token = string.(sort(collect(Set(text))))

    push!(id_to_token, "<s>")
    push!(id_to_token, "</s>")

    token_to_id = Dict(token => i for (i, token) in enumerate(id_to_token))

    return CharTokenizer(
        id_to_token,
        token_to_id,
        length(id_to_token) - 1,
        length(id_to_token),
    )
end

function encode(text::AbstractString, tokenizer::CharTokenizer)
    return [tokenizer.token_to_id[token] for token in text]
end
