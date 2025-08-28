import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset

from helpers.helpers import calculate_layer_entropy


MODEL_NAME = "google/gemma-2-2b"
MODEL_PATH = "./models/gemma-2-baseline.pth"
TOKENIZER_PATH = "./models/tokenizer"
DATA_CACHE_DIR = "./data"
N_PROMPTS_TO_SCAN = 1000
MAX_SEQ_LEN = 1024
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

print("\n--- Starting Analysis Loop ---")
results_list = []
layers_to_analyze = [0, 1, 2, 5, 10, 15, 20, 24, 25] 
with torch.no_grad():
    for i, example in enumerate(exploratory_set):
        if i >= N_PROMPTS_TO_SCAN:
            break

        prompt_text = example['text']
        print(f"\n=======================================================")
        print(f"--- Analyzing Prompt #{i+1} ---")
        print(f"Prompt: '{prompt_text[:100]}...'")
        print(f"=======================================================")

        tokens = hooked_model.to_tokens(prompt_text)
        truncated_tokens = tokens[:, :MAX_SEQ_LEN]
        _, cache = hooked_model.run_with_cache(truncated_tokens)

        prompt_results = {
            'prompt_index': i,
            'prompt_start': prompt_text[:100],
            'entropies': {}
        }

        for target_layer in layers_to_analyze:
            activations = cache[f"blocks.{target_layer}.hook_resid_post"]   # final output of the entire decoder block at layer 'target_layer'
            average_entropy = calculate_layer_entropy(hooked_model, activations)
            prompt_results['entropies'][target_layer] = average_entropy


        results_list.append(prompt_results)
        print(f"Prompt #{i+1} Entropies (by layer): {prompt_results['entropies']}")

        del cache
        del activations
        torch.cuda.empty_cache()

print("\n--- Analysis Complete ---")

results_df = pd.json_normalize(results_list)
results_df.to_csv("./entropy.csv")
print("\n--- Results Summary ---")
print(results_df.head())