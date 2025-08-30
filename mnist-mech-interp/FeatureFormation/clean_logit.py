import torch
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"

# === CIRCUIT COMPONENT TO TEST ===
TARGET_LAYER = 23
TARGET_HEAD = 0
HYPOTHESIS = "Head L23H0 is a 'Let's go' propagator"
# =================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
print("--- Loading model ---")
tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)
state_dict = torch.load("./models/gemma-2-baseline.pth", weights_only=True)
model.load_state_dict(state_dict)
model.eval()
hooked_model = HookedTransformer.from_pretrained(
    MODEL_NAME, hf_model=model, tokenizer=tokenizer, device=device, torch_dtype=torch.bfloat16
)
hooked_model.eval()
print("--- Model loaded successfully ---")

# --- Experiment Data ---
# Using a clear, representative example for the chart
patching_examples_df = pd.read_csv("./results/circuit_results/lets-go/causal_intervention_results_patching.csv")
examples = []
for index, row in patching_examples_df.iterrows():
    # Safely evaluate the string representation of the tuple
    answer_pair = eval(row['answer_pair'])
    examples.append((row['clean_prompt'], row['corrupted_prompt'], answer_pair))

# --- Global variables for hooks ---
clean_head_output_store = None
clean_answer_token_store = None
corrupted_answer_token_store = None

# --- Causal Intervention Functions (defined once) ---
def calculate_logit_diff(logits):
    """Calculates the difference in logits between the clean and corrupted answer tokens."""
    last_token_logits = logits[0, -1, :]
    clean_logit = last_token_logits[clean_answer_token_store]
    corrupted_logit = last_token_logits[corrupted_answer_token_store]
    return (clean_logit - corrupted_logit).item()

def patch_head_z_hook(head_output, hook):
    """Hook function to overwrite a specific head's output vector (z)."""
    head_output[0, -1, TARGET_HEAD, :] = clean_head_output_store
    return head_output

# --- Main Experiment Loop ---
all_results = []
hook_name = utils.get_act_name("z", TARGET_LAYER)

with torch.no_grad():
    for clean_prompt, corrupted_prompt, answer_pair in examples:
        
        # Update globals with current example's tokens
        clean_answer_token_store = hooked_model.to_tokens(f" {answer_pair[0]}", prepend_bos=False).item()
        corrupted_answer_token_store = hooked_model.to_tokens(f" {answer_pair[1]}", prepend_bos=False).item()

        clean_tokens = hooked_model.to_tokens(clean_prompt)
        corrupted_tokens = hooked_model.to_tokens(corrupted_prompt)

        print(f"\n--- Running Patching for L{TARGET_LAYER}H{TARGET_HEAD} on prompts ending with '{answer_pair[0]}' vs '{answer_pair[1]}' ---")

        # STEP 1: Run the clean prompt to get baseline logits AND cache the activation
        clean_logits, clean_cache = hooked_model.run_with_cache(clean_tokens)
        clean_logit_diff = calculate_logit_diff(clean_logits) # <-- KEY ADDITION
        clean_head_output_store = clean_cache[hook_name][0, -1, TARGET_HEAD, :]
        print(f"Clean logit diff: {clean_logit_diff:.4f}")

        # STEP 2: Run the corrupted prompt for the baseline corrupted result
        corrupted_logits = hooked_model(corrupted_tokens)
        corrupted_logit_diff = calculate_logit_diff(corrupted_logits)
        print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

        # STEP 3: Run the corrupted prompt with the patch
        patched_logits = hooked_model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_head_z_hook)]
        )
        patched_logit_diff = calculate_logit_diff(patched_logits)
        print(f"Patched logit diff: {patched_logit_diff:.4f}")

        # STEP 4: Measure and store the results
        causal_effect = patched_logit_diff - corrupted_logit_diff
        print(f"Causal effect of patch: {causal_effect:.4f}")

        all_results.append({
            'layer': TARGET_LAYER,
            'head': TARGET_HEAD,
            'hypothesis': HYPOTHESIS,
            'clean_prompt': clean_prompt,
            'corrupted_prompt': corrupted_prompt,
            'answer_pair': str(answer_pair),
            'clean_logit_diff': clean_logit_diff, # <-- SAVING THE NEW VALUE
            'corrupted_logit_diff': corrupted_logit_diff,
            'patched_logit_diff': patched_logit_diff,
            'causal_effect': causal_effect
        })

# --- Aggregate and Display Average Results ---
results_df = pd.DataFrame(all_results)

# Calculate the average of the logit diffs across all examples
avg_clean_logit_diff = results_df['clean_logit_diff'].mean()
avg_corrupted_logit_diff = results_df['corrupted_logit_diff'].mean()
avg_patched_logit_diff = results_df['patched_logit_diff'].mean()
avg_causal_effect = results_df['causal_effect'].mean()

print("\n" + "="*50)
print("--- Average Results Across All Examples ---")
print(f"Average Clean Logit Diff:      {avg_clean_logit_diff:.4f}")
print(f"Average Corrupted Logit Diff:  {avg_corrupted_logit_diff:.4f}")
print(f"Average Patched Logit Diff:    {avg_patched_logit_diff:.4f}")
print(f"Average Causal Effect:         {avg_causal_effect:.4f}")
print("="*50)