import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset

MODEL_NAME = "google/gemma-2-2b"
MODEL_PATH = "./models/gemma-2-baseline.pth"
TOKENIZER_PATH = "./models/tokenizer"
DATA_CACHE_DIR = "./data"
N_PROMPTS_TO_SCAN = 1000
MAX_SEQ_LEN = 1024
SPARSITY_THRESHOLD = 0.01
TOP_K_NEURONS = 30
SEED = 42

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

full_dataset = load_dataset("roneneldan/TinyStories", split='validation', cache_dir=DATA_CACHE_DIR)
shuffled_dataset = full_dataset.shuffle(seed=SEED)
split_data = shuffled_dataset.train_test_split(test_size=0.1, seed=SEED)
exploratory_set = split_data['train']
hold_out_set = split_data['test']


print(f"--- Starting Phase 1: Scanning {N_PROMPTS_TO_SCAN} prompts to gather neuron stats ---")

n_layers = hooked_model.cfg.n_layers
d_mlp = hooked_model.cfg.d_mlp
neuron_stats = [
    {'total_count': 0, 'dead_count': 0} 
    for _ in range(n_layers * d_mlp)
]

with torch.no_grad():
    for i, example in enumerate(exploratory_set):
        if i >= N_PROMPTS_TO_SCAN:
            break
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{N_PROMPTS_TO_SCAN} prompts...")

        tokens = hooked_model.to_tokens(example['text'])
        truncated_tokens = tokens[:, :MAX_SEQ_LEN]
        _, cache = hooked_model.run_with_cache(truncated_tokens)
        
        seq_len = truncated_tokens.shape[1]

        for layer_idx in range(n_layers):
            # Shape: [1, seq_len, d_mlp]
            activations = cache[f"blocks.{layer_idx}.mlp.hook_post"]
            
            # Count how many activations for each neuron in this layer are below the threshold
            # Shape: [d_mlp]
            dead_in_this_prompt = torch.sum(activations.squeeze(0) < SPARSITY_THRESHOLD, dim=0)

            for neuron_idx in range(d_mlp):
                # Flatten the (layer, neuron) index to a single list index
                flat_idx = layer_idx * d_mlp + neuron_idx
                
                neuron_stats[flat_idx]['total_count'] += seq_len
                neuron_stats[flat_idx]['dead_count'] += dead_in_this_prompt[neuron_idx].item()

        # Essential memory management
        del cache
        del activations
        torch.cuda.empty_cache()

print("--- Phase 1 Complete ---")

print("\n--- Starting Phase 2: Scoring and ranking all neurons ---")
ranked_neurons = []
with torch.no_grad():
    for layer_idx in range(n_layers):
        # Get the output weights for this layer's MLP
        # W_out has shape [d_mlp, d_model]
        W_out = hooked_model.blocks[layer_idx].mlp.W_out
        
        for neuron_idx in range(d_mlp):
            flat_idx = layer_idx * d_mlp + neuron_idx
            stats = neuron_stats[flat_idx]

            if stats['total_count'] > 0:
                # 1. Sparsity Score: % of the time the neuron was "dead"
                sparsity = stats['dead_count'] / stats['total_count']
                
                # 2. Importance Score: The norm of the neuron's output weights.
                # A neuron with larger output weights has a bigger "voice".
                # We take the norm of the row corresponding to our neuron.
                importance = torch.linalg.norm(W_out[neuron_idx, :]).item()
                
                # 3. Final Score: Combine them (we are looking for specialists - neurons that are influential and selectivly fire)
                final_score = sparsity * importance
                
                ranked_neurons.append({
                    'score': final_score,
                    'layer': layer_idx,
                    'neuron': neuron_idx,
                    'sparsity': sparsity,
                    'importance': importance
                })

# Sort the list to find the top neurons
ranked_neurons.sort(key=lambda x: x['score'], reverse=True)
df = pd.DataFrame(ranked_neurons)
print(df.head().to_string())
df.to_csv("./neuron_rankings.csv", index=False)

print("--- Phase 2 Complete ---")

# --- Display Results ---
print(f"\n--- Top {TOP_K_NEURONS} Most Promising Neurons (ranked by Sparsity * Importance) ---")
print("="*80)
print(f"{'Rank':<5} | {'Layer':<7} | {'Neuron':<8} | {'Score':<15} | {'Sparsity':<15} | {'Importance':<15}")
print("-"*80)

for rank, data in enumerate(ranked_neurons[:TOP_K_NEURONS]):
    print(f"{rank+1:<5} | {data['layer']:<7} | {data['neuron']:<8} | {data['score']:<15.4f} | {data['sparsity']:<15.4f} | {data['importance']:<15.4f}")

print("="*80)
print("\nAnalysis complete. To investigate a neuron, use the mini-test from Technique 2")
print(f"For example: TARGET_LAYER = {ranked_neurons[0]['layer']}, TARGET_NEURON = {ranked_neurons[0]['neuron']}")