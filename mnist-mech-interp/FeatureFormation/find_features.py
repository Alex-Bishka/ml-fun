import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"
DATA_CACHE_DIR = "./data"
SEED = 42
N_PROMPTS_TO_SCAN = 1000  # Scan more prompts for better results
TOP_K_EXAMPLES = 30       # How many top examples to display
CONTEXT_WINDOW = 10       # Tokens of context to show on each side of the trigger token

# === TARGET NEURON TO INVESTIGATE ===
# Let's start with the #1 ranked neuron from your table
TARGET_LAYER = 0
TARGET_NEURON = 1827
# ====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model and Data Loading (assuming model/data is already downloaded) ---
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

print(f"--- Starting Scan for Neuron L{TARGET_LAYER}-N{TARGET_NEURON} ---")

top_examples = []

with torch.no_grad():
    for i, example in enumerate(tqdm(exploratory_set, desc="Scanning Prompts")):
        if i >= N_PROMPTS_TO_SCAN:
            break

        tokens = hooked_model.to_tokens(example['text'])
        
        _, cache = hooked_model.run_with_cache(tokens)
        
        # Get activations for our specific neuron
        # Shape: [seq_len]
        neuron_activations = cache[f"blocks.{TARGET_LAYER}.mlp.hook_post"][0, :, TARGET_NEURON]
        
        # Find the max activation in this sequence
        max_activation, max_idx = torch.max(neuron_activations, dim=0)
        
        # Store the activation value, the token tensor, and the position of the max activation
        top_examples.append({
            'activation': max_activation.item(),
            'tokens': tokens,
            'position': max_idx.item()
        })

# --- Process and Display Results ---
print("\n--- Processing and displaying top examples ---")

# Sort all examples by activation value in descending order
top_examples.sort(key=lambda x: x['activation'], reverse=True)

# Get just the top K examples
top_examples = top_examples[:TOP_K_EXAMPLES]

print("="*80)
print(f"Top {len(top_examples)} Activating Examples for Neuron L{TARGET_LAYER}-N{TARGET_NEURON}")
print("="*80)

for i, example in enumerate(top_examples):
    activation_value = example['activation']
    tokens = example['tokens']
    position = example['position']
    
    # Extract the trigger token and the context around it
    trigger_token = hooked_model.to_str_tokens(tokens[:, position])[0]
    
    start = max(0, position - CONTEXT_WINDOW)
    end = min(tokens.shape[1], position + CONTEXT_WINDOW + 1)
    context_str_list = hooked_model.to_str_tokens(tokens[:, start:end])
    relative_position = position - start
    context_str_list[relative_position] = f"-->{context_str_list[relative_position]}<--"
    highlighted_context = "".join(context_str_list)

    print(f"{i+1:2d}) Activation: {activation_value:<8.4f} | Trigger: '{trigger_token}'")
    print(f"   Context: ...{highlighted_context}...")
    print("-"*80)