# TODO: clean up and add a mock model
# Note: developed on model Stories15M
# using Llama2: str_lookup, bpe_encode

# @testset "str_lookup" begin
#     @test str_lookup(vocab[2], vocab, config.vocab_size) == 2
#     pos = str_lookup("š", vocab, config.vocab_size)
#     @test "š" == vocab[pos] |> copy |> String
# end

# @testset "bpe_encode" begin
#     (; config, weights) = model
#     (; vocab, vocab_scores) = model.tokenizer

#     toks = bpe_encode(" g", vocab, vocab_scores, config.vocab_size)
#     @test vocab[toks[1]] |> copy |> String == " g"

#     toks = bpe_encode("and", vocab, vocab_scores, config.vocab_size)
#     @test vocab[toks[1]] == codeunits(s)

#     toks = bpe_encode("Juλia", vocab, vocab_scores, config.vocab_size)
#     @test toks == [78, 121, 30143, 424]

#     toks = bpe_encode("grabbed", vocab, vocab_scores, vocab_size)
#     @test vocab[toks[1]] |> copy |> String == "gra"
#     @test vocab[toks[2]] |> copy |> String == "bb"
#     @test vocab[toks[3]] |> copy |> String == "ed"
# end
