using Llama2

# Run AK model
sample("models/model.bin",
    "models/tokenizer.bin";
    temperature = 0.0f0)

model = load_model("models/model.bin", "models/tokenizer.bin");
sample(model, "Julia is the best"; temperature = 0.5f0)