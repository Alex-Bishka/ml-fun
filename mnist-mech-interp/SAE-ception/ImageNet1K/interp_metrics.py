import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mutual_info_score

# ACTIVATIONS_BASE_PATH = "./SAE-Results/test-features/baseline/act-baseline"
# ACTIVATIONS_BASE_PATH = "./act-F0"
ACTIVATIONS_BASE_PATH = "./act-F1"

def get_sorted_chunks(path):
    """Finds and sorts chunk files numerically."""
    chunk_files = glob.glob(os.path.join(path, 'chunk_*.npz'))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {path}")
    
    sort_key = lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])
    return sorted(chunk_files, key=sort_key)

all_sparse_codes = []
all_labels = []
chunk_files = get_sorted_chunks(ACTIVATIONS_BASE_PATH)
for chunk_path in tqdm(chunk_files, desc="Analyzing Chunks"):
    with np.load(chunk_path) as data:
        # all_sparse_codes.append(data['hidden'])
        all_sparse_codes.append(data['sparse'])
        all_labels.append(data['labels'])

codes = np.vstack(all_sparse_codes)
labels = np.concatenate(all_labels)

H = codes.shape[1]
mi_indices = np.zeros(H)
csi_indices = np.zeros(H)

unique_classes = np.unique(labels)
num_classes = len(unique_classes)
for j in range(H):
    act_j = codes[:, j]

    # normalized MI variant
    thr = np.percentile(act_j, 75)           # e.g. threshold at top 25%
    binarized = (act_j > thr).astype(int)
    mi = mutual_info_score(binarized, labels)
    p_on = binarized.mean()
    h_on = -(p_on*np.log2(p_on + 1e-12) + (1-p_on)*np.log2(1-p_on + 1e-12))
    mi_indices[j] = mi / (h_on + 1e-12)

    # standard CSI
    act_j_rectified = np.maximum(0, act_j)
    class_means = np.array([np.mean(act_j_rectified[labels == c]) for c in unique_classes])
    mu_max = np.max(class_means)
    max_idx = np.argmax(class_means)
    mu_other = np.mean(class_means[np.arange(num_classes) != max_idx]) if num_classes > 1 else 0
    denominator = mu_max + mu_other + 1e-12
    csi_indices[j] = (mu_max - mu_other) / denominator if denominator != 0 else 0

print("Mean class-selectivity (normalized MI):", mi_indices.mean())
print("Mean class-selectivity:", csi_indices.mean())
print("#" * 60)



############################################################
# Baseline:
# Mean class-selectivity (normalized MI): 0.2750986783758622
# Mean class-selectivity: 
############################################################
# F0:
# Mean class-selectivity (normalized MI): 0.28255749796
# Mean class-selectivity: 0.9929022545111366
############################################################
# F1:
############################################################
# F2:
############################################################