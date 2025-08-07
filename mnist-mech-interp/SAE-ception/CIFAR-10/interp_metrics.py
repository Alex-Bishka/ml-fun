

from sklearn.metrics import mutual_info_score

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

codes = activation_data['sparse']
labels = activation_data['labels']

H = codes.shape[1]
cs_indices = np.zeros(H)
for j in range(H):
    # binarize activation on atom j
    act_j = codes[:, j]
    thr = np.percentile(act_j, 75)           # e.g. threshold at top 25%
    binarized = (act_j > thr).astype(int)

    # mutual info between “atom-on/off” and class label
    mi = mutual_info_score(binarized, labels)

    # normalize by entropy of the on/off signal
    p_on = binarized.mean()
    h_on = -(p_on*np.log2(p_on + 1e-12) + (1-p_on)*np.log2(1-p_on + 1e-12))
    cs_indices[j] = mi / (h_on + 1e-12)

print("Mean class-selectivity (normalized MI):", cs_indices.mean())
print("#" * 60)



############################################################
# Baseline:
# Mean class-selectivity (normalized MI): 0.05164640227130353
############################################################
# F0:
# Mean class-selectivity (normalized MI): 0.04432501377537306
############################################################
# F1:
# Mean class-selectivity (normalized MI): 0.04956580502943177
############################################################