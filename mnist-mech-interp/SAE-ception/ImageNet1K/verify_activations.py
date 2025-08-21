import numpy as np
from datasets import load_dataset

# Import your own set_seed function to ensure consistency
from helpers.helpers import set_seed 

# --- CRITICAL ---
# These constants MUST be IDENTICAL to the ones in your
# gen_targets.py and adapt_classifier.py scripts.
SEED = 42
VAL_SET_SIZE = 25000
# ACTIVATIONS_BASE_PATH_c = "./SAE-Results/training-features/baseline/activations"
# ACTIVATIONS_BASE_PATH_0 = "./SAE-Results/training-features/F0/activations"
ACTIVATIONS_BASE_PATH_c = "./activations-baseline"
ACTIVATIONS_BASE_PATH_0 = "./activations"


chunk_path_c = f"{ACTIVATIONS_BASE_PATH_c}/chunk_0.npz"
chunk_data_c = np.load(chunk_path_c)
chunk_path_0 = f"{ACTIVATIONS_BASE_PATH_0}/chunk_0.npz"
chunk_data_0 = np.load(chunk_path_0)
# The labels are saved as a numpy array in the chunk
for i in range(len(chunk_data_c["labels"])):
    chunk_data_c['sparse'][i] 
    chunk_data_0['sparse'][i]

    if not np.array_equal(chunk_data_0, chunk_data_c):
        print(f"-> NOT the same!")
    continue

import sys
sys.exit(0)
print("--- Running Activation Verification Script ---")

# --- Step 1: Recreate the exact dataset split from your scripts ---
print(f"Using SEED={SEED} and VAL_SET_SIZE={VAL_SET_SIZE} to recreate the dataset split...")
set_seed(SEED)

full_train_split = load_dataset(
    "ILSVRC/imagenet-1k", 
    split='train', 
    cache_dir='./data',
    trust_remote_code=True  # Suppress the warning
)
shuffled_train = full_train_split.shuffle(seed=SEED)
train_split = shuffled_train.select(range(VAL_SET_SIZE, len(shuffled_train)))

# Get the label that *should* be at the very first position
expected_sample = train_split[0]
expected_label = expected_sample['label']

print(f"\n[EXPECTED] The label for the first sample in the code should be: {expected_label}")


# --- Step 2: Load the first chunk file from disk and check its first label ---
chunk_path = f"{ACTIVATIONS_BASE_PATH}/chunk_0.npz"

try:
    print(f"Loading existing chunk file: {chunk_path}")
    chunk_data = np.load(chunk_path)
    # The labels are saved as a numpy array in the chunk
    for i in range(len(chunk_data["labels"])):
        actual_label = chunk_data['labels'][i]
        expected_sample = train_split[i]
        expected_label = expected_sample['label']
        if int(expected_label) != int(actual_label):
            print(i, expected_label, actual_label)
        continue
        print(f"[ACTUAL]   The label for the first sample in the chunk file is: {actual_label}")

        # --- Step 3: Compare and give a definitive result ---
        print("\n--- Verification Result ---")
        if int(expected_label) == int(actual_label):
            print("✅ MATCH: The chunk file appears to be in sync with your code.")
            print("The error may lie elsewhere, but the files are not stale.")
        else:
            print("❌ MISMATCH: The chunk file on disk is STALE and out of sync with your code.")
            print("This confirms that you need to delete and regenerate your activation files.")
            print(i, expected_label, actual_label)
        print("---------------------------\n")

except FileNotFoundError:
    print(f"\nERROR: Could not find the chunk file at '{chunk_path}'.")
    print("Please ensure you have generated the activations at least once.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")