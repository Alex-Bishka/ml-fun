import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score

from helpers.helpers import SNE_plot_2d

# ACTIVATIONS_BASE_PATH = "./SAE-Results/test-features/baseline/activations"
# ACTIVATIONS_BASE_PATH = "./SAE-Results/test-features/F0/activations"
# ACTIVATIONS_BASE_PATH = "./act-baseline"
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
        all_sparse_codes.append(data['sparse'])
        all_labels.append(data['labels'])

sparse_codes = np.vstack(all_sparse_codes)
labels = np.concatenate(all_labels)
max_feature_indices_sparse_codes = np.argmax(sparse_codes, axis=1)

print("Calculating cluster metrics")
X = sparse_codes
true_labels = labels
pred_labels = max_feature_indices_sparse_codes

ari = adjusted_rand_score(true_labels, pred_labels)
sil_unsup = silhouette_score(X,          pred_labels)   # how tight are your feature-induced clusters?
sil_sup   = silhouette_score(X,          true_labels)   # how well do features separate the real digits?
db        = davies_bouldin_score(X,      true_labels)   # with true labels
ch        = calinski_harabasz_score(X,   true_labels)

metrics_df = pd.DataFrame({
    'Metric': [
        'Adjusted Rand Index (ARI)',
        'Silhouette (Unsupervised)',
        'Silhouette (Supervised)',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Index'
    ],
    'Value': [ari, sil_unsup, sil_sup, db, ch]
})

metrics_df['Value'] = metrics_df['Value'].round(3)
metrics_df.to_csv(f"./metrics.csv")
print(metrics_df)

# print("Generating TSNE plot")
# tsne = TSNE(n_components=2, random_state=42)
# activations_2d = tsne.fit_transform(sparse_codes)

# SNE_plot_2d(activations_2d=activations_2d, labels=labels, 
            # cluster_labels=max_feature_indices_sparse_codes, height=800, width=800)