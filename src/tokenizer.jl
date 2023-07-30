struct Tokenizer
    vocab::Vector{String}
    vocab_scores::Vector{Float32}
end

function load_tokenizer(filename::AbstractString, vocab_size::Int)
    vocab = Vector{String}(undef, vocab_size)
    vocab_scores = Vector{Float32}(undef, vocab_size)

    open(filename) do file
        max_token_length = read(file, Int32)
        for i in 1:vocab_size
            vocab_scores[i] = read(file, Float32)
            len = read(file, Int32)
            vocab[i] = String(read(file, len))
        end
    end

    return Tokenizer(vocab, vocab_scores)
end

function bpe_encode(text::AbstractString, tokenizer::Tokenizer)
    (; vocab, vocab_scores) = tokenizer

    # encode every individual char in the input string
    # this throws if the char is not in vocab
    tokens = Int[findfirst(str -> length(str) == 1 && str[1] == char, vocab) for char in text]

    # a temporary buffer to merge two consecutive tokens
    str_buffer = UInt8[]

    while true
        best_score = -Inf32
        best_id = -1
        best_idx = -1

        for i in 1:(length(tokens) - 1)
            # check if we can merge the pair (tokens[i], tokens[i+1])
            empty!(str_buffer)
            append!(str_buffer, codeunits(vocab[tokens[i]]))
            append!(str_buffer, codeunits(vocab[tokens[i+1]]))

            id = findfirst(str -> codeunits(str) == str_buffer, vocab)

            if !isnothing(id) && vocab_scores[id] > best_score
                best_score = vocab_scores[id]
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
