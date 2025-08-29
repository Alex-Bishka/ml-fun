import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"

# === LAYER TO TEST ===
TARGET_LAYER = 25
HYPOTHESIS = "The Declarative feature is encoded in the MLP of Layer 25"
# ======================

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

# --- Experiment Setup ---
clean_prompt = "The captain stood bravely on the deck of the"
corrupted_prompt = "On the deck stood bravely the captain of the"
answer_pair = ('ship', 'sea') # clean first, corrupt second

clean_answer_token = hooked_model.to_tokens(f" {answer_pair[0]}", prepend_bos=False).item()
corrupted_answer_token = hooked_model.to_tokens(f" {answer_pair[1]}", prepend_bos=False).item()

clean_tokens = hooked_model.to_tokens(clean_prompt)
corrupted_tokens = hooked_model.to_tokens(corrupted_prompt)

# The expectation is highest at the end, so we will patch at the final token position.
patch_position = -1

# The hook point is the output of the entire MLP block
hook_name = f"blocks.{TARGET_LAYER}.mlp.hook_post"


# --- Causal Intervention Functions ---
def calculate_logit_diff(logits):
    """Calculates the difference in logits between the clean and corrupted answer tokens."""
    last_token_logits = logits[0, -1, :]
    clean_logit = last_token_logits[clean_answer_token]
    corrupted_logit = last_token_logits[corrupted_answer_token]
    return (clean_logit - corrupted_logit).item()

clean_mlp_output = None # Global to be set in the experiment
def patch_mlp_output(mlp_output, hook):
    """
    Hook function to overwrite the ENTIRE MLP output vector at a specific position.
    mlp_output shape: [batch, position, d_model]
    """
    mlp_output[0, patch_position, :] = clean_mlp_output
    return mlp_output


# --- Running the Experiment ---
with torch.no_grad():
    print(f"\n--- Running MLP Layer Patching Experiment for Layer {TARGET_LAYER} ---")
    
    # STEP 1: Run the clean prompt and cache the MLP output vector
    _, clean_cache = hooked_model.run_with_cache(clean_tokens)
    clean_mlp_output = clean_cache[hook_name][0, patch_position, :]
    print(f"Clean MLP output vector captured from Layer {TARGET_LAYER}")

    # STEP 2: Run the corrupted prompt to get our baseline result.
    corrupted_logits = hooked_model(corrupted_tokens)
    corrupted_logit_diff = calculate_logit_diff(corrupted_logits)
    print(f"Original (corrupted) logit diff ('{answer_pair[0]}' - '{answer_pair[1]}'): {corrupted_logit_diff:.4f}")

    # STEP 3: Run the corrupted prompt AGAIN, but patch in the clean MLP output.
    patched_logits = hooked_model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, patch_mlp_output)]
    )
    patched_logit_diff = calculate_logit_diff(patched_logits)
    print(f"Patched (corrupted w/ clean MLP output) logit diff: {patched_logit_diff:.4f}")

    # STEP 4: Measure the causal effect.
    causal_effect = patched_logit_diff - corrupted_logit_diff
    print("\n--- Results ---")
    print(f"Causal effect of patching the MLP layer: {causal_effect:.4f}")

# --- Save the results to a CSV log ---
outcome = "Hypothesis Confirmed" if causal_effect > 0.1 else "Hypothesis Falsified / Weak Effect"
experiment_data = {
    'layer': TARGET_LAYER,
    'neuron': 'MLP_LAYER', # Note that we're patching the whole layer
    'hypothesis': HYPOTHESIS,
    'clean_prompt': clean_prompt,
    'corrupted_prompt': corrupted_prompt,
    'answer_pair': str(answer_pair),
    'clean_activation': 'N/A (MLP Output)',
    'corrupted_logit_diff': corrupted_logit_diff,
    'patched_logit_diff': patched_logit_diff,
    'causal_effect': causal_effect,
    'outcome': outcome
}

results_df = pd.DataFrame([experiment_data])
output_csv_path = 'causal_intervention_results.csv'

print(f"\n--- Saving results for L{experiment_data['layer']} MLP ---")
if os.path.exists(output_csv_path):
    results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    print(f"Appended new results to '{output_csv_path}'")
else:
    results_df.to_csv(output_csv_path, mode='w', header=True, index=False)
    print(f"Created new results log at '{output_csv_path}'")