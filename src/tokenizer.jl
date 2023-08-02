struct Tokenizer
    id_to_token::Vector{String}
    token_to_id::Dict{String,Int}
    token_scores::Vector{Float32}
end

function bpe_encode(text::AbstractString, tokenizer::Tokenizer; pad_input=true)
    (; id_to_token, token_to_id, token_scores) = tokenizer

    tokens = Int[]

    if isempty(text)
        return tokens
    end

    if pad_input
        push!(tokens, token_to_id[" "])
    end

    # encode every individual char in the input string
    for char in text
        push!(tokens, token_to_id[string(char)])
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
