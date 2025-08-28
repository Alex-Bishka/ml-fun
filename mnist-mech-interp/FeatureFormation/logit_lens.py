import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm # For a nice progress bar

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"
DATA_CACHE_DIR = "./data"
SEED = 42
N_PROMPTS_TO_ANALYZE = 1000
MAX_SEQ_LEN = 1024
TOP_K_TOKENS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model and Data Loading (same as before) ---
print("--- Loading model and data ---")
tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
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

full_dataset = load_dataset("roneneldan/TinyStories", split='validation', cache_dir=DATA_CACHE_DIR)
shuffled_dataset = full_dataset.shuffle(seed=SEED)
split_data = shuffled_dataset.train_test_split(test_size=0.1, seed=SEED)
exploratory_set = split_data['train']

print("--- Model and data loaded successfully ---")


# --- Main Analysis Loop ---
print(f"\n--- Starting Analysis: Calculating average logits for top {TOP_K_TOKENS} tokens per layer across {N_PROMPTS_TO_ANALYZE} prompts ---")

# Layers to analyze (early, mid, late)
layers_to_analyze = [0, 1, 2, 5, 10, 15, 20, 24, 25] 


logit_sums = {layer: torch.zeros(hooked_model.cfg.d_vocab, device=device) for layer in layers_to_analyze}
token_counts = {layer: 0 for layer in layers_to_analyze}
with torch.no_grad():
    for i, example in enumerate(tqdm(exploratory_set, desc="Processing Prompts")):
        if i >= N_PROMPTS_TO_ANALYZE:
            break

        prompt_text = example['text']
        tokens = hooked_model.to_tokens(prompt_text)
        truncated_tokens = tokens[:, :MAX_SEQ_LEN]
        _, cache = hooked_model.run_with_cache(truncated_tokens)
        
        W_U = hooked_model.W_U 
        for target_layer in layers_to_analyze:
            activations = cache[f"blocks.{target_layer}.hook_resid_post"]
            layer_logits = activations @ W_U
            summed_logits_for_prompt = layer_logits.squeeze(0).sum(dim=0)
            logit_sums[target_layer] += summed_logits_for_prompt
            token_counts[target_layer] += tokens.shape[1]

# --- Post-processing and DataFrame Creation ---
print("\n--- Aggregating results and creating DataFrame ---")

results_list = []
for layer in layers_to_analyze:
    if token_counts[layer] == 0:
        continue

    avg_logits = logit_sums[layer] / token_counts[layer]
    top_k_values, top_k_indices = torch.topk(avg_logits, TOP_K_TOKENS)
    top_k_tokens = hooked_model.to_str_tokens(top_k_indices)
    
    for rank, (token, value) in enumerate(zip(top_k_tokens, top_k_values)):
        results_list.append({
            'layer': layer,
            'rank': rank + 1,
            'token': token,
            'avg_logit': value.item()
        })

# Create the DataFrame
df_layer_preds = pd.DataFrame(results_list)

# Save the DataFrame to a CSV file
output_csv_path = "./average_layer_predictions.csv"
df_layer_preds.to_csv(output_csv_path, index=False)

print(f"\n--- Analysis Complete ---")
print(f"Results saved to {output_csv_path}")

# Display a pivot table for easy comparison
print("\n--- Top 5 Average Token Predictions per Layer ---")
pivot_df = df_layer_preds[df_layer_preds['rank'] <= 5].pivot_table(
    index='rank', 
    columns='layer', 
    values='token', 
    aggfunc='first'
)
print(pivot_df.to_string())