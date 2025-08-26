import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset

# --- 1. SETUP (Assuming you've run this) ---
MODEL_NAME = "google/gemma-2-2b"
DATA_CACHE_DIR = "./data"
SEED = 42

# Ensure you have a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "google/gemma-2-2b-it"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, # <-- Note: we use the original name for the architecture
    torch_dtype=torch.bfloat16
)
state_dict = torch.load("./models/gemma-2-baseline.pth", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

hooked_model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    hf_model=model,
    tokenizer=tokenizer,
    device=device,
    torch_dtype=torch.bfloat16
)

# Load and split the dataset
full_dataset = load_dataset("roneneldan/TinyStories", split='validation', cache_dir=DATA_CACHE_DIR)
shuffled_dataset = full_dataset.shuffle(seed=SEED)
exploratory_set = shuffled_dataset 


# --- 2. THE ANALYSIS LOOP ---
print("\n--- Starting Analysis Loop ---")
for i, example in enumerate(exploratory_set):
    if i >= 3:
        break

    prompt_text = example['text']
    print(f"\n--- Analyzing Prompt #{i+1} ---")
    print(f"Prompt: '{prompt_text[:100]}...'")

    # Tokenize the prompt. run_with_cache wants token IDs.
    tokens = hooked_model.to_tokens(prompt_text) # Shape: [1, sequence_length]
    
    # --- This is the core of TransformerLens ---
    # Run the model and cache all intermediate activations
    # We set stop_at_layer to not compute the final logits, just for this example
    logits, cache = hooked_model.run_with_cache(tokens)

    # `cache` is a dictionary-like object holding all the activations.
    # Let's inspect the residual stream after layer 8
    target_layer = 8
    # The hook name follows a standard format.
    # 'blocks.{layer_index}.hook_resid_post' gets the output of the residual stream
    layer_8_activations = cache[f"blocks.{target_layer}.hook_resid_post"]
    
    # The shape is [batch_size, sequence_length, d_model]
    # In our case, batch_size is 1.
    print(f"Shape of activations at Layer {target_layer}: {layer_8_activations.shape}")

    # --- 3. APPLYING THE LOGIT LENS ---
    # Project the activations from layer 8 back into the vocabulary space
    # by multiplying with the unembedding matrix (W_U).
    
    # Get the unembedding matrix from the model
    W_U = hooked_model.W_U # Shape: [d_model, d_vocab]
    
    # Project activations to logits
    # Resulting shape: [batch_size, seq_len, d_model] @ [d_model, d_vocab] -> [batch_size, seq_len, d_vocab]
    layer_8_logits = layer_8_activations @ W_U

    # Get the top 5 predicted tokens at each position in the sequence
    top_k_values, top_k_indices = torch.topk(layer_8_logits, 5, dim=-1)

    # Let's look at the model's predictions at the 10th token position
    position_to_inspect = 10
    if tokens.shape[1] > position_to_inspect:
      original_token_str = hooked_model.to_str_tokens(tokens[:, position_to_inspect])[0]
      print(f"\n--- Logit Lens at Layer {target_layer}, Position {position_to_inspect} (Original Token: '{original_token_str}') ---")

      # Decode the top k token predictions
      top_k_tokens_at_pos = hooked_model.to_str_tokens(top_k_indices[:, position_to_inspect, :].squeeze(0))

      for token, value in zip(top_k_tokens_at_pos, top_k_values[:, position_to_inspect, :].squeeze(0)):
          print(f"Prediction: '{token}' (Logit: {value:.2f})")

print("\n--- Analysis Complete ---")