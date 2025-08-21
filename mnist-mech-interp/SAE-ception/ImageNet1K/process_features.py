import os
import numpy as np
import torch
import pickle
import glob
from tqdm import tqdm

from helpers.sae import SparseAutoencoder # Make sure this import is correct

# --- 1. Configuration ---
FEATURE_DIM = 1536
SPARSE_DIM = FEATURE_DIM * 4 # The dimension of the SAE's hidden layer
NUM_CLASSES = 1000 # ImageNet-1k has 1000 classes, not 10
N_TOP_FEATURES = 25 # The 'N' for finding top features

# Match these with your training/generation scripts
# SAE_MODEL_PATH = "./sae_models/baseline/sae_last_layer_l1_0.0005_30.pth"
# SAE_MODEL_PATH = "./sae_models/F0/sae_last_layer_l1_0.0002.pth"
SAE_MODEL_PATH = "./sae_models/F1/sae_last_layer_l1_0.0001.pth"

# ACTIVATIONS_BASE_PATH = "./SAE-Results/training-features/baseline/activations"
# ACTIVATIONS_BASE_PATH = "./SAE-Results/training-features/F0/act-F0"
ACTIVATIONS_BASE_PATH = "act-F1" 

OUTPUT_PATH = f"./aux-activations-top-{N_TOP_FEATURES}"
TOP_FEATURES_FILE = "./top_N_features.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Helper function to sort chunk files numerically ---
def get_sorted_chunks(path):
    """Finds and sorts chunk files numerically."""
    chunk_files = glob.glob(os.path.join(path, 'chunk_*.npz'))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {path}")
    
    sort_key = lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])
    return sorted(chunk_files, key=sort_key)

# --- 3. Phase 1: Analyze all chunks to find top features per class ---
def calculate_top_features():
    """
    Iterates through all activation chunks to find the average sparse activation
    for each class, then determines the top N features.
    """
    print("--- Phase 1: Calculating Top N Features ---")
    
    # Initialize accumulators for streaming calculation
    # We need to store the sum of sparse vectors and a count for each class
    sum_of_activations = np.zeros((NUM_CLASSES, SPARSE_DIM), dtype=np.float32)
    count_of_samples = np.zeros(NUM_CLASSES, dtype=np.int32)
    
    chunk_files = get_sorted_chunks(ACTIVATIONS_BASE_PATH)
    
    for chunk_path in tqdm(chunk_files, desc="Analyzing Chunks"):
        with np.load(chunk_path) as data:
            sparse_act = data['sparse']
            labels = data['labels']
            
            # Efficiently add activations to the sums for each class present in the chunk
            for class_idx in np.unique(labels):
                mask = (labels == class_idx)
                sum_of_activations[class_idx] += np.sum(sparse_act[mask], axis=0)
                count_of_samples[class_idx] += np.sum(mask)

    print("Calculating class averages...")
    # Calculate the average activation for each class, avoiding division by zero
    # for classes that might not have samples (unlikely in ImageNet)
    avg_class_encoding = np.zeros_like(sum_of_activations)
    valid_counts = count_of_samples > 0
    avg_class_encoding[valid_counts] = sum_of_activations[valid_counts] / count_of_samples[valid_counts, np.newaxis]

    print(f"Finding top {N_TOP_FEATURES} features for each class...")
    top_n_features = {}
    for class_idx in range(NUM_CLASSES):
        # Sort features by average activation strength in descending order
        top_n_indices = np.argsort(avg_class_encoding[class_idx])[-N_TOP_FEATURES:][::-1]
        top_n_features[class_idx] = {'indices': top_n_indices}
        
    print(f"Saving top feature indices to {TOP_FEATURES_FILE}")
    with open(TOP_FEATURES_FILE, "wb") as f:
        pickle.dump(top_n_features, f)
        
    return top_n_features

# --- 4. Phase 2: Generate new reconstructions using the top features ---
def generate_aux_reconstructions(top_n_features):
    """
    Iterates through all activation chunks again, using the top_n_features
    to generate and save new chunked reconstructions.
    """
    print("\n--- Phase 2: Generating Auxiliary Reconstructions ---")

    print("Loading SAE model...")
    sae = SparseAutoencoder(input_dim=FEATURE_DIM).to(device)
    sae.load_state_dict(torch.load(SAE_MODEL_PATH))
    sae.eval()
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    chunk_files = get_sorted_chunks(ACTIVATIONS_BASE_PATH)

    for chunk_idx, chunk_path in enumerate(tqdm(chunk_files, desc="Generating New Chunks")):
        with np.load(chunk_path) as data:
            sparse_act = data['sparse']
            labels = data['labels']
            
            # Prepare batch of encoded vectors for efficient GPU processing
            batch_size = len(labels)
            encoded_top = np.zeros((batch_size, SPARSE_DIM), dtype=np.float32)
            encoded_ablated = sparse_act.copy() # Start with the full sparse vector

            for i in range(batch_size):
                label = labels[i]
                if label in top_n_features:
                    feature_indices = top_n_features[label]['indices']
                    # Create vector with ONLY top features
                    encoded_top[i, feature_indices] = sparse_act[i, feature_indices]
                    # Create ablated vector by zeroing out top features
                    encoded_ablated[i, feature_indices] = 0
            
            # Process the entire chunk in one go on the GPU
            with torch.no_grad(), torch.amp.autocast('cuda'):
                encoded_top_torch = torch.from_numpy(encoded_top).to(device)
                encoded_ablated_torch = torch.from_numpy(encoded_ablated).to(device)
                
                recon_top = sae.decoder(encoded_top_torch).cpu().numpy()
                recon_ablated = sae.decoder(encoded_ablated_torch).cpu().numpy()

            # Save the new chunk with the generated reconstructions and original labels
            output_chunk_path = os.path.join(OUTPUT_PATH, f"aux_chunk_{chunk_idx}.npz")
            np.savez_compressed(
                output_chunk_path,
                recon_top=recon_top,
                recon_ablated=recon_ablated,
                labels=labels
            )

    print(f"\n✅ Done. Auxiliary reconstructions saved in '{OUTPUT_PATH}'")


if __name__ == "__main__":
    # Run Phase 1 to get the feature dictionary
    top_features = calculate_top_features()
    
    # Run Phase 2 using the results of Phase 1
    generate_aux_reconstructions(top_features)