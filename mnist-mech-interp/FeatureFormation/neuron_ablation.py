import torch
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"

# === NEURON TO TEST ===
TARGET_LAYER = 3
TARGET_NEURON = 3576
AMPLIFICATION_FACTOR = 2.0  # How much to boost the activation in the amplification experiment
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
prompt = ""
answer_pair = ('Once', 'Therefore')  # clean first, corrupt second

clean_answer_token = hooked_model.to_tokens(f" {answer_pair[0]}", prepend_bos=False).item()
corrupted_answer_token = hooked_model.to_tokens(f" {answer_pair[1]}", prepend_bos=False).item()

tokens = hooked_model.to_tokens(prompt, prepend_bos=True)

# Find the specific token position for our noun
noun_token_id = hooked_model.to_single_token(" jellyfish")
# token_position = (tokens == noun_token_id).nonzero()[0, 1].item()
token_position = 0

hook_name = f"blocks.{TARGET_LAYER}.mlp.hook_post"


# --- Causal Intervention Functions ---
def calculate_logit_diff(logits):
    # last_token_logits = logits[0, -1, :]
    last_token_logits = logits[0, 0, :]
    clean_logit = last_token_logits[clean_answer_token]
    corrupted_logit = last_token_logits[corrupted_answer_token]
    return (clean_logit - corrupted_logit).item()

# Hooks for our two experiments
def ablation_hook(activation, hook):
    # Set the neuron's activation to zero at the target position
    activation[0, token_position, TARGET_NEURON] = 0.0
    return activation

natural_activation_value = None # Global variable to hold the original activation
def amplification_hook(activation, hook):
    # Multiply the neuron's natural activation by our factor
    activation[0, token_position, TARGET_NEURON] = natural_activation_value * AMPLIFICATION_FACTOR
    return activation


# --- Running the Experiments ---
with torch.no_grad():
    print(f"\n--- Running Ablation & Amplification for Neuron L{TARGET_LAYER}-N{TARGET_NEURON} ---")
    
    # STEP 1: Baseline run to get the natural behavior and activation value
    baseline_logits, baseline_cache = hooked_model.run_with_cache(tokens)
    natural_activation_value = baseline_cache[hook_name][0, token_position, TARGET_NEURON].item()
    baseline_logit_diff = calculate_logit_diff(baseline_logits)
    
    # STEP 2: Ablation run (zeroing out the neuron)
    ablated_logits = hooked_model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, ablation_hook)]
    )
    ablated_logit_diff = calculate_logit_diff(ablated_logits)
    
    # STEP 3: Amplification run (boosting the neuron)
    amplified_logits = hooked_model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, amplification_hook)]
    )
    amplified_logit_diff = calculate_logit_diff(amplified_logits)

    # STEP 4: Calculate effects
    ablation_effect = ablated_logit_diff - baseline_logit_diff
    amplification_effect = amplified_logit_diff - baseline_logit_diff

    print("\n--- Results ---")
    print(f"Natural activation (at 'parrot'): {natural_activation_value:.4f}")
    print(f"Baseline logit diff {answer_pair}: {baseline_logit_diff:.4f}")
    print("-" * 20)
    print(f"Ablated logit diff: {ablated_logit_diff:.4f}")
    print(f"Causal effect of ABLATION: {ablation_effect:.4f}")
    print("-" * 20)
    print(f"Amplified logit diff (x{AMPLIFICATION_FACTOR}): {amplified_logit_diff:.4f}")
    print(f"Causal effect of AMPLIFICATION: {amplification_effect:.4f}")


# --- Save both results to the CSV log ---
ablation_data = {
    'layer': TARGET_LAYER, 'neuron': TARGET_NEURON, 'hypothesis': 'Ablation damages concept of activating noun',
    'clean_prompt': prompt, 'corrupted_prompt': 'N/A (Ablation)', 'answer_pair': str(answer_pair),
    'clean_activation': natural_activation_value, 'corrupted_logit_diff': baseline_logit_diff,
    'patched_logit_diff': ablated_logit_diff, 'causal_effect': ablation_effect,
    'outcome': "Hypothesis Confirmed" if ablation_effect < -0.1 else "Hypothesis Falsified / Weak Effect"
}
amplification_data = {
    'layer': TARGET_LAYER, 'neuron': TARGET_NEURON, 'hypothesis': 'Amplification enhances concept of activating noun',
    'clean_prompt': prompt, 'corrupted_prompt': f'N/A (Amplified x{AMPLIFICATION_FACTOR})', 'answer_pair': str(answer_pair),
    'clean_activation': natural_activation_value, 'corrupted_logit_diff': baseline_logit_diff,
    'patched_logit_diff': amplified_logit_diff, 'causal_effect': amplification_effect,
    'outcome': "Hypothesis Confirmed" if amplification_effect > 0.1 else "Hypothesis Falsified / Weak Effect"
}

results_df = pd.DataFrame([ablation_data, amplification_data])
output_csv_path = 'causal_intervention_results.csv'

print(f"\n--- Saving Ablation & Amplification results ---")
if os.path.exists(output_csv_path):
    results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    print(f"Appended new results to '{output_csv_path}'")
else:
    results_df.to_csv(output_csv_path, mode='w', header=True, index=False)
    print(f"Created new results log at '{output_csv_path}'")

# --- Display the current log ---
full_results_log = pd.read_csv(output_csv_path)
print("\n--- Current Causal Intervention Log ---")
print(full_results_log.tail(5).to_string()) # Display the last 5 experiments