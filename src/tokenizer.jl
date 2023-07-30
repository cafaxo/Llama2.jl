"""
    str_lookup(cu::AbstractVector{UInt8}, vocab::Vector{<:Vector{UInt8}}, vocab_size::Int)

Looks up a position of token represented by codeunits `cu` in `vocab`
"""
function str_lookup(cu::AbstractVector{UInt8},
    vocab::Vector{<:Vector{UInt8}},
    vocab_size::Int)
    # find the first perfect match for str in vocab, return its index or -1 if not found
    for i in 1:vocab_size
        if cu == vocab[i]
            return i
        end
    end
    return -1
end
function str_lookup(str::String, vocab::Vector{<:Vector{UInt8}}, vocab_size::Int)
    str_lookup(codeunits(str), vocab, vocab_size)
end

"""
    bpe_encode(text::String, vocab::Vector{<:Vector{UInt8}}, vocab_scores::Vector{Float64}, vocab_size::Int, max_token_length::Int, tokens::Vector{Int}, n_tokens::Int)

Encodes text into tokens based on `vocab` and `vocab_scores`

# Example

```julia
toks = bpe_encode("and", vocab, vocab_scores, config.vocab_size)
# [393] represents 393rd token in `vocab`
```
"""
function bpe_encode(text::String,
    vocab::Vector{<:Vector{UInt8}},
    vocab_scores::Vector{Float32},
    vocab_size::Int)
    tokens = Vector{Int}(undef, length(text))

    # a temporary buffer to merge two consecutive tokens
    str_buffer = ""

    # first encode every individual byte in the input string
    n_tokens = 0 # the number of tokens
    for c in text
        id = str_lookup(string(c), vocab, vocab_size)
        if id == -1
            error("Not found: $c")
        end
        tokens[n_tokens + 1] = id
        n_tokens += 1
    end

    # merge the best consecutive pair each iteration, according the scores in vocab_scores
    while true
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for i in 1:(n_tokens - 1)
            # check if we can merge the pair (tokens[i], tokens[i+1])
            str_buffer = vcat(vocab[tokens[i]], vocab[tokens[i + 1]])
            id = str_lookup(str_buffer, vocab, vocab_size)
            if id != -1 && vocab_scores[id] > best_score
                # this merge pair exists in vocab! record its score and position
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
        for i in (best_idx + 1):(n_tokens - 1)
            tokens[i] = tokens[i + 1]
        end
        n_tokens -= 1 # token length decreased
    end

    return tokens[1:n_tokens]
end
