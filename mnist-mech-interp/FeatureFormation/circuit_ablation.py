import torch
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils
from functools import partial

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"

# === CIRCUIT TO INTERVENE ON ===
# The heads we found with a causal effect > 1.0
HEADS_TO_INTERVENE = [
    (7, 4),
    (21, 6),
    (23, 0),
    (24, 3),
]
AMPLIFICATION_FACTOR = 2.0
# ===============================

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
examples = [
    # ("Come on, everyone! Let's", ('go', 'act'))

    ("The sun is shining! Let's", ('go', 'stay')),
    ("The movie is about to start, Let's", ('go', 'wait')),
    ("I'm hungry. Let's", ('go', 'eat')), # A slightly different context to test generalization

    # --- Prompts inspired by the CSV data ---
    ("The park is just around the corner. Let's", ('go', 'run')),
    ("I hear music downstairs. Let's", ('go', 'listen')),
    ("Look at that huge slide! Let's", ('go', 'jump')),

    # --- More complex sentence structures ---
    ("We've finished all our chores for the day. Now, let's", ('go', 'rest')),
    ("The treasure map says the 'X' is in the old cave. Come on, let's", ('go', 'look')),

    # --- A negative context to test robustness ---
    ("It's starting to rain. I don't think we should be out here. Let's", ('go', 'stay')),
    
    # --- A more abstract, less physical "go" ---
    ("We need to start this project. Let's",('go','begin'))
]
for prompt, answer_pair in examples:
    print(prompt)
    print(answer_pair)
    clean_answer_token = hooked_model.to_tokens(f" {answer_pair[0]}", prepend_bos=False).item()
    corrupted_answer_token = hooked_model.to_tokens(f" {answer_pair[1]}", prepend_bos=False).item()
    tokens = hooked_model.to_tokens(prompt)


    # --- Causal Intervention Functions ---
    def calculate_logit_diff(logits):
        last_token_logits = logits[0, -1, :]
        clean_logit = last_token_logits[clean_answer_token]
        corrupted_logit = last_token_logits[corrupted_answer_token]
        return (clean_logit - corrupted_logit).item()

    # Generic hooks for ablation and amplification
    def ablation_hook_for_head(head_output, hook, head_to_ablate):
        head_output[0, -1, head_to_ablate, :] = 0.0
        return head_output

    def amplification_hook_for_head(head_output, hook, head_to_amplify, natural_head_output):
        head_output[0, -1, head_to_amplify, :] = natural_head_output * AMPLIFICATION_FACTOR
        return head_output


    # --- Running the Experiments ---
    with torch.no_grad():
        print(f"\n--- Running Multi-Head Interventions on {len(HEADS_TO_INTERVENE)} heads ---")
        
        # STEP 1: Baseline run to get natural behavior and activations
        baseline_logits, baseline_cache = hooked_model.run_with_cache(tokens)
        baseline_logit_diff = calculate_logit_diff(baseline_logits)
        
        # STEP 2: Multi-Head Ablation
        ablation_hooks = []
        for layer_idx, head_idx in HEADS_TO_INTERVENE:
            hook_name = utils.get_act_name("z", layer_idx)
            # Use partial to "bake in" the head_idx for each hook
            hook_func = partial(ablation_hook_for_head, head_to_ablate=head_idx)
            ablation_hooks.append((hook_name, hook_func))
        
        ablated_logits = hooked_model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)
        ablated_logit_diff = calculate_logit_diff(ablated_logits)
        
        # STEP 3: Multi-Head Amplification
        amplification_hooks = []
        for layer_idx, head_idx in HEADS_TO_INTERVENE:
            hook_name = utils.get_act_name("z", layer_idx)
            natural_head_output = baseline_cache[hook_name][0, -1, head_idx, :]
            hook_func = partial(amplification_hook_for_head, head_to_amplify=head_idx, natural_head_output=natural_head_output)
            amplification_hooks.append((hook_name, hook_func))
            
        amplified_logits = hooked_model.run_with_hooks(tokens, fwd_hooks=amplification_hooks)
        amplified_logit_diff = calculate_logit_diff(amplified_logits)

        # STEP 4: Calculate effects
        ablation_effect = ablated_logit_diff - baseline_logit_diff
        amplification_effect = amplified_logit_diff - baseline_logit_diff

        print("\n--- Results ---")
        print(f"Baseline logit diff {answer_pair}: {baseline_logit_diff:.4f}")
        print("-" * 20)
        print(f"Multi-Head Ablated logit diff: {ablated_logit_diff:.4f}")
        print(f"Causal effect of ABLATING {len(HEADS_TO_INTERVENE)} heads: {ablation_effect:.4f}")
        print("-" * 20)
        print(f"Multi-Head Amplified logit diff (x{AMPLIFICATION_FACTOR}): {amplified_logit_diff:.4f}")
        print(f"Causal effect of AMPLIFYING {len(HEADS_TO_INTERVENE)} heads: {amplification_effect:.4f}")

    # --- Save both results to the CSV log ---
    ablation_data = {
        'layer': str([h[0] for h in HEADS_TO_INTERVENE]), 'neuron': str([h[1] for h in HEADS_TO_INTERVENE]), 
        'hypothesis': 'Multi-Head Ablation damages concept',
        'clean_prompt': prompt, 'corrupted_prompt': 'N/A (Multi-Ablation)', 'answer_pair': str(answer_pair),
        'clean_activation': 'N/A', 'corrupted_logit_diff': baseline_logit_diff,
        'patched_logit_diff': ablated_logit_diff, 'causal_effect': ablation_effect,
        'outcome': "Hypothesis Confirmed" if ablation_effect < -0.1 else "Hypothesis Falsified / Weak Effect"
    }
    amplification_data = {
        'layer': str([h[0] for h in HEADS_TO_INTERVENE]), 'neuron': str([h[1] for h in HEADS_TO_INTERVENE]), 
        'hypothesis': 'Multi-Head Amplification enhances concept',
        'clean_prompt': prompt, 'corrupted_prompt': f'N/A (Multi-Amplified x{AMPLIFICATION_FACTOR})', 'answer_pair': str(answer_pair),
        'clean_activation': 'N/A', 'corrupted_logit_diff': baseline_logit_diff,
        'patched_logit_diff': amplified_logit_diff, 'causal_effect': amplification_effect,
        'outcome': "Hypothesis Confirmed" if amplification_effect > 0.1 else "Hypothesis Falsified / Weak Effect"
    }

    results_df = pd.DataFrame([ablation_data, amplification_data])
    output_csv_path = 'causal_intervention_results.csv'
    print(f"\n--- Saving Multi-Head Intervention results ---")
    if os.path.exists(output_csv_path):
        results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
        print(f"Appended new results to '{output_csv_path}'")
    else:
        results_df.to_csv(output_csv_path, mode='w', header=True, index=False)
        print(f"Created new results log at '{output_csv_path}'")
    full_results_log = pd.read_csv(output_csv_path)
    print("\n--- Current Causal Intervention Log ---")
    print(full_results_log.tail(5).to_string())