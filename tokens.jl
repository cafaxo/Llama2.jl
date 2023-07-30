using Llama2

config = open("data_raw/stories15M.bin") do file
    config = Llama2.read_config(file)
end
# config = open("data_raw/model.bin") do file
#     config = Llama2.read_config(file)
# end

vocab = Vector{Vector{UInt8}}(undef, config.vocab_size)
vocab_scores = Vector{Float32}(undef, config.vocab_size)
max_token_length = 1

open("data_raw/tokenizer.bin") do file
    max_token_length = read(file, Int32)
    for i in 1:(config.vocab_size)
        vocab_scores[i] = read(file, Float32)
        len = read(file, Int32)
        vocab[i] = read(file, len)
    end
end;

# Run the benchmark
sample("models/model.bin",
    "models/tokenizer.bin";
    temperature = 0.0f0)

# custom prompt
x = 300
for i in x:(x + 10)
    @info "$i: $(vocab[i]|>copy|>String)"
end
s = "say hi!"
[codepoint(c) for c in s]
'š' |> codepoint
"š" |> codeunits

@assert str_lookup(vocab[2], vocab, config.vocab_size) == 2
pos = str_lookup("š", vocab, config.vocab_size)
@assert "š" == vocab[pos] |> copy |> String
codeunits("Juλia")

s = "and"
toks = bpe_encode(s, vocab, vocab_scores, config.vocab_size)
@assert vocab[toks[1]] == codeunits(s)

s = "say hi!"
toks = bpe_encode(s, vocab, vocab_scores, config.vocab_size)
@assert toks == [20835, 7252, 37]

s = "grabbed"
toks = bpe_encode(s, vocab, vocab_scores, config.vocab_size)
vocab[toks[1]] |> copy |> String
vocab[toks[2]] |> copy |> String
vocab[toks[3]] |> copy |> String

s = " g"
toks = bpe_encode(s, vocab, vocab_scores, config.vocab_size)
vocab[toks[1]] |> copy |> String
vocab[toks[1]][1]
" " |> codeunits

# 
# start with token 2 -- BOS
# following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
vocab[1] |> copy |> String

model = load_model("models/model.bin", "models/tokenizer.bin")
sample(model, "Julia is the best"; temperature = 0.5f0)

xxx
# <s>[INST] <<SYS>>
# {{ system_prompt }}
# <</SYS>>

# {{ user_message }} [/INST]

# // start the main loop
# long start = 0;  // used to time our code, only initialized after first iteration
# int next;        // will store the next token in the sequence
# int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
# int pos = 0;     // position in the sequence
# printf("<s>\n"); // explicit print the initial BOS token for stylistic symmetry reasons
# while (pos < steps) {

#     // forward the transformer to get logits for the next token
#     transformer(token, pos, &config, &state, &weights);

#     if(pos < num_prompt_tokens) {
#         // if we are still processing the input prompt, force the next prompt token
#         next = prompt_tokens[pos];
#     } else {
#         // sample the next token
#         if (temperature == 0.0f) {
#             // greedy argmax sampling: take the token with the highest probability
#             next = argmax(state.logits, config.vocab_size);
#         } else {
#             // apply the temperature to the logits
#             for (int q=0; q<config.vocab_size; q++) { state.logits[q] /= temperature; }
#             // apply softmax to the logits to get the probabilities for next token
#             softmax(state.logits, config.vocab_size);
#             // we sample from this distribution to get the next token
#             next = sample(state.logits, config.vocab_size);
#         }
#     }

#     // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
#     char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next]+1 : vocab[next];
#     printf("%s", token_str);
#     fflush(stdout);

#     // advance forward
#     token = next;
#     pos++;
#     // init our timer here because the first iteration is slow due to memmap
#     if (start == 0) { start = time_in_ms(); }
# }
