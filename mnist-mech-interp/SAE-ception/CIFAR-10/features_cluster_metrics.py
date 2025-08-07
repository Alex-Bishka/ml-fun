import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE

from helpers.sae import SparseAutoencoder
from helpers.helpers import extract_activations, SNE_plot_2d

EMBEDS_PATH = './embeds/pos_embed_edge_384_99.56.pth'
# VIT_PATH = './classifiers/baseline/vit_h_99.56.pth'
# VIT_PATH = './classifiers/F0/vit_h_99.56_25_top_0.0002_99.41.pth'
VIT_PATH = './classifiers/F1/best_model_lf_0.01.pth'
# SAE_PATH = './sae_models/baseline-99.56/last_layer/sae_last_layer_l1_0.0002.pth'
# SAE_PATH = './sae_models/F0/sae_last_layer_l1_0.0002.pth'
SAE_PATH = './sae_models/F1/sae_last_layer_l1_0.0002.pth'
IMG_RES = 384
FEATURE_DIM = 1280
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"We will be using device: {device}")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_RES, IMG_RES)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                pin_memory=True, num_workers=4)

print("Loading ViT")
weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
model = vit_h_14(weights=weights) 
model.image_size = IMG_RES  # Update the expected image size

model.encoder.pos_embedding = torch.nn.Parameter(torch.load(EMBEDS_PATH))

num_ftrs = model.heads.head.in_features
model.heads.head = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(VIT_PATH))
model.to(device)

print("Loading SAE")
sae = SparseAutoencoder(input_dim=FEATURE_DIM)
sae.load_state_dict(torch.load(SAE_PATH))
sae.to(device)

print("Extracting activations")
activation_data = extract_activations(
    data_loader=test_loader,
    model=model,
    sae=sae,
    device=device
)

sparse_codes = activation_data["sparse"]
labels = activation_data['labels']
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

print("Generating TSNE plot")
tsne = TSNE(n_components=2, random_state=42)
activations_2d = tsne.fit_transform(sparse_codes)

SNE_plot_2d(activations_2d=activations_2d, labels=labels, 
            cluster_labels=max_feature_indices_sparse_codes, height=800, width=800)