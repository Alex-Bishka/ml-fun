import torch
import pandas as pd
import os
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "google/gemma-2-2b"

# === CIRCUIT TO DISCOVER ===
# We are tracing the output of our Noun Detector in Layer 3
START_LAYER = 1 # Start scanning layers AFTER the neuron
# =========================

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
clean_prompt = "Come on, everyone! Let's"
corrupted_prompt = "Come on, everyone! We must"
answer_pair = ('go', 'act')

clean_answer_token = hooked_model.to_tokens(f" {answer_pair[0]}", prepend_bos=False).item()
corrupted_answer_token = hooked_model.to_tokens(f" {answer_pair[1]}", prepend_bos=False).item()

clean_tokens = hooked_model.to_tokens(clean_prompt)
corrupted_tokens = hooked_model.to_tokens(corrupted_prompt)

# --- Causal Intervention Functions ---
def calculate_logit_diff(logits):
    last_token_logits = logits[0, -1, :]
    clean_logit = last_token_logits[clean_answer_token]
    corrupted_logit = last_token_logits[corrupted_answer_token]
    return (clean_logit - corrupted_logit).item()


# --- Running the Experiment ---
with torch.no_grad():
    print("\n--- Running Path Patching Experiment to Find Circuit ---")
    
    corrupted_logits = hooked_model(corrupted_tokens)
    corrupted_logit_diff = calculate_logit_diff(corrupted_logits)
    print(f"Original (corrupted) logit diff {answer_pair}: {corrupted_logit_diff:.4f}")

    _, clean_cache = hooked_model.run_with_cache(clean_tokens)
    
    n_layers = hooked_model.cfg.n_layers
    n_heads = hooked_model.cfg.n_heads
    
    results = []
    
    pbar = tqdm(total=(n_layers - START_LAYER) * n_heads, desc="Patching Heads")

    for layer_idx in range(START_LAYER, n_layers):
        for head_idx in range(n_heads):
            hook_name = utils.get_act_name("z", layer_idx)
            
            clean_head_output = clean_cache[hook_name][0, -1, head_idx, :]
            
            # --- NEW: Define the hook function inside the loop (a closure) ---
            def patch_head_z_hook(head_output, hook):
                # This function now has access to head_idx from the loop
                head_output[0, -1, head_idx, :] = clean_head_output
                return head_output
            # -----------------------------------------------------------------

            # Run the corrupted prompt with the dynamically created hook
            patched_logits = hooked_model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, patch_head_z_hook)] # No more hook_ctx
            )
            
            patched_logit_diff = calculate_logit_diff(patched_logits)
            causal_effect = patched_logit_diff - corrupted_logit_diff
            
            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'causal_effect': causal_effect
            })
            pbar.update(1)
            
    pbar.close()

# --- Analyze and Display Results ---
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='causal_effect', ascending=False)

print("\n--- Top 10 Most Influential Attention Heads ---")
print(results_df.head(10).to_string())

# --- Create and Save Heatmap ---
print("\n--- Generating Heatmap of Head Causal Effects ---")
heatmap_data = results_df.pivot(index="layer", columns="head", values="causal_effect")
fig = px.imshow(
    heatmap_data,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    labels=dict(x="Head Index", y="Layer", color="Causal Effect"),
    title="Causal Effect of Patching Each Attention Head's Output (hook_z)"
)

output_heatmap_path = "head_patching_heatmap.html"
fig.write_html(output_heatmap_path)
print(f"Interactive heatmap saved to: {output_heatmap_path}")