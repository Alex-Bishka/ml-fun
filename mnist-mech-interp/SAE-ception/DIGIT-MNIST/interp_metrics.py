from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import numpy as np
from models_and_data.nn import NeuralNetwork
from models_and_data.sae import SparseAutoencoder
from models_and_data.edgedataset import EdgeDataset
from models_and_data.model_helpers import (load_intermediate_labels, extract_activations, evaluate_and_gather_activations)

from sklearn.metrics import mutual_info_score

HIDDEN_SIZE = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"We will be using device: {device}")

# train data
train_images = load_intermediate_labels("./intermediate-labels/first_layer/train_images.pkl")
train_labels = load_intermediate_labels("./intermediate-labels/first_layer/train_labels.pkl")

# test data
test_images = load_intermediate_labels("./intermediate-labels/first_layer/test_images.pkl")
test_labels = load_intermediate_labels("./intermediate-labels/first_layer/test_labels.pkl")

seed = 42
generator = torch.Generator().manual_seed(seed)

NUM_WORKERS = 4
if device.type.lower() == "cpu":
    NUM_WORKERS = 0

# test data
test_dataset = EdgeDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

layer = 'one'
model_paths = [
    # "./SAE-Results/256-0.75/results/baseline/model_state_dict.pth",
    # "./SAE-Results/256-0.75/results/F0/models/25_top/best_model_lf_0.14.pth",
    # "./SAE-Results/256-0.75/results/F1/models/25_top_0.14/25_top/best_model_lf_0.06.pth",
    # "./SAE-Results/256-0.75/results/F2/models/25_top_0.14_25_top_0.06/25_top/best_model_lf_0.18.pth"

    "./models_saved/256_mask/best_model_lf_0.29.pth"

    # "./SAE-Results/256-0.75/results/baseline/model_state_dict.pth",
    # "./SAE-Results/256-0.75/results/F0/models/25_mask/best_model_lf_0.01.pth",
    # "./SAE-Results/256-0.75/results/F1/models/25_mask_0.01/25_mask/best_model_lf_0.06.pth",
    # "./SAE-Results/256-0.75/results/F2/models/25_mask_0.01_25_mask_0.06/25_mask/best_model_lf_0.2.pth"

    # "./SAE-Results/256-0.75/results/baseline/model_state_dict.pth",
    # "./SAE-Results/256-0.75/results/F0/models/256_top/best_model_lf_0.07.pth",
    # "./SAE-Results/256-0.75/results/F1/models/256_top_0.07/256_top/best_model_lf_0.04.pth",
    # "./SAE-Results/256-0.75/results/F2/models/256_top_0.07_256_top_0.04/256_top/best_model_lf_0.03.pth"

    # "./SAE-Results/256-0.75/results/baseline/model_state_dict.pth",
    # "./SAE-Results/256-0.75/results/F0/models/256_mask/best_model_lf_0.29.pth",
    # "./SAE-Results/256-0.75/results/F1/models/256_mask_0.29/256_mask/best_model_lf_0.02.pth",
    # "./SAE-Results/256-0.75/results/F2/models/256_mask_0.29_256_mask_0.02/256_mask/best_model_lf_0.13.pth"

    # "./full-total_epoch_100/full-34.pth",
    # "./full-total_epoch_100/full-70.pth"
]
for best_model_path in model_paths:
    print(best_model_path)
    checkpoint = torch.load(best_model_path)

    model = NeuralNetwork().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    sae = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)
    sae.load_state_dict(checkpoint[f'sae_{layer}_state_dict'])

    test_results = evaluate_and_gather_activations(model, sae, sae, test_loader, device)
    print(f"Model acc: {test_results['accuracy']}")

    activation_data = extract_activations(
        data_loader=test_loader,
        model=model,
        sae_one=sae,
        sae_two=sae,
        device=device
    )

    # codes = activation_data[f'sparse_{layer}']
    codes = activation_data[f'hidden_{layer}']
    labels = activation_data['labels']


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