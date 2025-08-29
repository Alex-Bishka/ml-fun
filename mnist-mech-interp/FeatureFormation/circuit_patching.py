import torch
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"

# === CIRCUIT COMPONENT TO TEST ===
# Testing our single most important head from the circuit discovery
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


# --- Experiment Setup ---
examples = [
    (
        "The gate is open now. Let's",             # Clean prompt (induces 'go')
        "The book is open now. Let's",             # Corrupted prompt (induces 'read')
        ('go', 'read')                             # (clean_token, corrupted_token)
    ),

    # Example 6: Contrast between urgency/departure and stillness/quiet
    (
        "Hurry, the train is here! Let's",          # Clean prompt (induces 'go')
        "Quiet, the baby is asleep. Let's",        # Corrupted prompt (induces 'whisper' or 'be')
        ('go', 'whisper')                          # (clean_token, corrupted_token)
    ),

    # Example 7: Contrast between physical action and mental action
    (
        "The path ahead looks clear. Let's",       # Clean prompt (induces 'go')
        "The problem ahead looks hard. Let's",     # Corrupted prompt (induces 'think')
        ('go', 'think')                            # (clean_token, corrupted_token)
    ),
    
    # Example 8: Contrast between starting an activity vs. ending one
    (
        "The concert is starting! Let's",          # Clean prompt (induces 'go')
        "The long day is ending. Let's",           # Corrupted prompt (induces 'rest')
        ('go', 'rest')                             # (clean_token, corrupted_token)
    ),

    # Example 9: Minimal pair with a different object implying a different verb
    (
        "Grab your coat and let's",                # Clean prompt (induces 'go')
        "Grab your fork and let's",                # Corrupted prompt (induces 'eat')
        ('go', 'eat')                              # (clean_token, corrupted_token)
    ),

    # Example 10: Simple context switch from outside to inside
    (
        "The taxi is waiting outside. Let's",      # Clean prompt (induces 'go')
        "The movie is starting inside. Let's",     # Corrupted prompt (induces 'watch')
        ('go', 'watch')                            # (clean_token, corrupted_token)
    ),
    # (
    #     "The coast is clear now. Let's",             # Clean prompt (induces 'go')
    #     "The instructions say to wait. We should",  # Corrupted prompt (induces 'wait')
    #     ('go', 'wait')                               # (clean_token, corrupted_token)
    # ),

    # # Example 2: Using a more abstract or nonsensical corrupted prompt
    # (
    #     "The adventure is this way! Let's",          # Clean prompt (induces 'go')
    #     "Tippy-tappy, toe-toe, on the",              # Corrupted prompt (induces something else)
    #     ('go', 'floor')                              # (clean_token, corrupted_token)
    # ),

    # # Example 3: Similar grammatical structure, but opposite semantic meaning
    # (
    #     "The bus is about to leave! Let's",         # Clean prompt (induces 'go')
    #     "The show is over, time to sit and",        # Corrupted prompt (induces 'rest')
    #     ('go', 'rest')                              # (clean_token, corrupted_token)
    # ),
    
    # # Example 4: A prompt that encourages a non-action verb
    # (
    #     "Everyone is waiting for us. Let's",         # Clean prompt (induces 'go')
    #     "After that long day, I think we should just", # Corrupted prompt (induces 'relax')
    #     ('go', 'relax')                              # (clean_token, corrupted_token)
    # ),
]
for clean_prompt, corrupted_prompt, answer_pair in examples:
    clean_answer_token = hooked_model.to_tokens(f" {answer_pair[0]}", prepend_bos=False).item()
    corrupted_answer_token = hooked_model.to_tokens(f" {answer_pair[1]}", prepend_bos=False).item()

    clean_tokens = hooked_model.to_tokens(clean_prompt)
    corrupted_tokens = hooked_model.to_tokens(corrupted_prompt)

    hook_name = utils.get_act_name("z", TARGET_LAYER)
    patch_position = -1 # Patching at the final token position


    # --- Causal Intervention Functions ---
    def calculate_logit_diff(logits):
        last_token_logits = logits[0, -1, :]
        clean_logit = last_token_logits[clean_answer_token]
        corrupted_logit = last_token_logits[corrupted_answer_token]
        return (clean_logit - corrupted_logit).item()

    clean_head_output = None # Global variable to store the clean head output
    def patch_head_z_hook(head_output, hook):
        # head_output (z) shape: [batch, pos, head_index, d_head]
        head_output[0, patch_position, TARGET_HEAD, :] = clean_head_output
        return head_output


    # --- Running the Experiment ---
    with torch.no_grad():
        print(f"\n--- Running Targeted Path Patching for Head L{TARGET_LAYER}H{TARGET_HEAD} ---")

        # STEP 1: Run the clean prompt and cache the activation of our target head
        _, clean_cache = hooked_model.run_with_cache(clean_tokens)
        clean_head_output = clean_cache[hook_name][0, patch_position, TARGET_HEAD, :]
        print(f"Clean output for L{TARGET_LAYER}H{TARGET_HEAD} captured.")

        # STEP 2: Run the corrupted prompt to get our baseline result.
        corrupted_logits = hooked_model(corrupted_tokens)
        corrupted_logit_diff = calculate_logit_diff(corrupted_logits)
        print(f"Original (corrupted) logit diff {answer_pair}: {corrupted_logit_diff:.4f}")

        # STEP 3: Run the corrupted prompt AGAIN, but this time, patch in the clean head output.
        patched_logits = hooked_model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_head_z_hook)]
        )
        patched_logit_diff = calculate_logit_diff(patched_logits)
        print(f"Patched (corrupted w/ clean head output) logit diff: {patched_logit_diff:.4f}")

        # STEP 4: Measure the causal effect.
        causal_effect = patched_logit_diff - corrupted_logit_diff
        print("\n--- Results ---")
        print(f"Causal effect of patching the head: {causal_effect:.4f}")


    # --- Save the results to a CSV log ---
    outcome = "Hypothesis Confirmed" if causal_effect > 0.1 else "Hypothesis Falsified / Weak Effect"
    experiment_data = {
        'layer': TARGET_LAYER,
        'neuron': TARGET_HEAD, # Using 'neuron' column for 'head' index for simplicity
        'hypothesis': HYPOTHESIS,
        'clean_prompt': clean_prompt,
        'corrupted_prompt': corrupted_prompt,
        'answer_pair': str(answer_pair),
        'clean_activation': 'N/A (Head Output)',
        'corrupted_logit_diff': corrupted_logit_diff,
        'patched_logit_diff': patched_logit_diff,
        'causal_effect': causal_effect,
        'outcome': outcome
    }

    results_df = pd.DataFrame([experiment_data])
    output_csv_path = 'causal_intervention_results_patching.csv'

    print(f"\n--- Saving results for L{experiment_data['layer']}H{experiment_data['neuron']} ---")
    if os.path.exists(output_csv_path):
        results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
        print(f"Appended new results to '{output_csv_path}'")
    else:
        results_df.to_csv(output_csv_path, mode='w', header=True, index=False)
        print(f"Created new results log at '{output_csv_path}'")

    # --- Display the current log ---
    full_results_log = pd.read_csv(output_csv_path)
    print("\n--- Current Causal Intervention Log ---")
    print(full_results_log.tail(5).to_string())