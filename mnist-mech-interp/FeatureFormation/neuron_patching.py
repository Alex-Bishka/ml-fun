import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"

# === NEURON TO TEST ===
TARGET_LAYER = 25
TARGET_NEURON = 8830

# --- Experiment Setup ---
HYPOTHESIS = "Sentence opener"
clean_prompt = "The captain stood bravely on the deck of the"
corrupted_prompt = "On the deck stood bravely the captain of the"
answer_pair = ('ship', 'sea') # clean first, corrupt second
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
    MODEL_NAME,
    hf_model=model,
    tokenizer=tokenizer,
    device=device,
    torch_dtype=torch.bfloat16
)
hooked_model.eval()
print("--- Model loaded successfully ---")

# Get the token IDs for our answer tokens
clean_answer_token = hooked_model.to_tokens(f" {answer_pair[0]}", prepend_bos=False).item()
corrupted_answer_token = hooked_model.to_tokens(f" {answer_pair[1]}", prepend_bos=False).item()

clean_tokens = hooked_model.to_tokens(clean_prompt)
corrupted_tokens = hooked_model.to_tokens(corrupted_prompt)

# --- CORRECTED TOKEN POSITIONING ---
# Find the specific token positions for our new single-token nouns
clean_token_id = hooked_model.to_single_token("The") # Note the leading space
# clean_token_position = -1
clean_token_position = (clean_tokens == clean_token_id).nonzero()[0, 1].item()

corrupted_token_id = hooked_model.to_single_token("On") # Note the leading space
# corrupted_token_position = -1
corrupted_token_position = (corrupted_tokens == corrupted_token_id).nonzero()[0, 1].item()

hook_name = f"blocks.{TARGET_LAYER}.mlp.hook_post"

# --- Causal Intervention Functions ---
def calculate_logit_diff(logits):
    last_token_logits = logits[0, -1, :]
    clean_logit = last_token_logits[clean_answer_token]
    corrupted_logit = last_token_logits[corrupted_answer_token]
    
    return (clean_logit - corrupted_logit).item()


def patch_neuron_activation(activation, hook):
    activation[0, corrupted_token_position, TARGET_NEURON] = clean_activation_value
    return activation


# --- Running the Experiment ---
with torch.no_grad():
    print(f"\n--- Running Causal Experiment for Neuron L{TARGET_LAYER}N{TARGET_NEURON} ---")
    
    # STEP 1: Run the clean prompt and cache the activation of our target neuron
    # at the final token position.
    _, clean_cache = hooked_model.run_with_cache(clean_tokens)
    clean_activation_value = clean_cache[hook_name][0, clean_token_position, TARGET_NEURON]
    print(f"Activation of L{TARGET_LAYER}N{TARGET_NEURON} on clean prompt: {clean_activation_value:.4f}")

    # STEP 2: Run the corrupted prompt to get our baseline result.
    corrupted_logits = hooked_model(corrupted_tokens)
    corrupted_logit_diff = calculate_logit_diff(corrupted_logits)
    print(f"Original (corrupted) logit diff ('{answer_pair[0]}' - '{answer_pair[1]}'): {corrupted_logit_diff:.4f}")

    # STEP 3: Run the corrupted prompt AGAIN, but this time, patch in the clean activation.
    # We use run_with_hooks to add our temporary patching hook.
    patched_logits = hooked_model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, patch_neuron_activation)]
    )
    patched_logit_diff = calculate_logit_diff(patched_logits)
    print(f"Patched (corrupted w/ clean activation) logit diff: {patched_logit_diff:.4f}")

    # STEP 4: Measure the causal effect.
    causal_effect = patched_logit_diff - corrupted_logit_diff
    print("\n--- Results ---")
    print(f"Causal effect of patching the neuron: {causal_effect:.4f}")

    if causal_effect > 0:
        print("\nSUCCESS!")
        print(f"This provides strong evidence that this neuron is CAUSAL for {HYPOTHESIS}.")
    else:
        print("\nExperiment did not show a positive causal effect. The neuron might be correlational or the effect is more subtle.")

outcome = "Hypothesis Confirmed" if causal_effect > 0.1 else "Hypothesis Falsified / Weak Effect" # Setting a small threshold
experiment_data = {
    'layer': TARGET_LAYER,
    'neuron': TARGET_NEURON,
    'hypothesis': HYPOTHESIS,
    'clean_prompt': clean_prompt,
    'corrupted_prompt': corrupted_prompt,
    'answer_pair': str(answer_pair),
    'clean_activation': clean_activation_value,
    'corrupted_logit_diff': corrupted_logit_diff,
    'patched_logit_diff': patched_logit_diff,
    'causal_effect': causal_effect,
    'outcome': outcome
}

results_df = pd.DataFrame([experiment_data])
output_csv_path = f'./results/neuron_causal_intervention/L{TARGET_LAYER}_N{TARGET_NEURON}.csv'

if os.path.exists(output_csv_path):
    results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    print(f"Appended new results to '{output_csv_path}'")
else:
    results_df.to_csv(output_csv_path, mode='w', header=True, index=False)
    print(f"Created new results log at '{output_csv_path}'")