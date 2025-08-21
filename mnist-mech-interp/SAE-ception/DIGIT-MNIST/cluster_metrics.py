HIDDEN_SIZE = 256

import os
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score

import pandas as pd

import torch
from torch.utils.data import DataLoader

from models_and_data.nn import NeuralNetwork
from models_and_data.sae import SparseAutoencoder
from models_and_data.edgedataset import EdgeDataset

from models_and_data.model_helpers import (evaluate_and_gather_activations, extract_activations, 
                                            load_intermediate_labels, seed_worker)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"We will be using device: {device}")

# train data
train_images = load_intermediate_labels("./intermediate-labels/first_layer/train_images.pkl")
train_labels = load_intermediate_labels("./intermediate-labels/first_layer/train_labels.pkl")

# test data
test_images = load_intermediate_labels("./intermediate-labels/first_layer/test_images.pkl")
test_labels = load_intermediate_labels("./intermediate-labels/first_layer/test_labels.pkl")

sparse_type = 'full'
mask_type = '256_top'
model_paths = [
    ("./models_saved/256_mask/best_model_lf_0.29.pth", ".")
    # ("./SAE-Results/256-0.75/results/baseline/model_state_dict.pth", f"./interp/{sparse_type}/baseline"),
    # (f"./SAE-Results/256-0.75/results/F0/models/{mask_type}/best_model_lf_0.07.pth",
    #  f"./interp/{sparse_type}/F0"),
    # (f"./SAE-Results/256-0.75/results/F1/models/{mask_type}_0.07/{mask_type}/best_model_lf_0.04.pth",
    #  f"./interp/{sparse_type}/F1"),
    # (f"./SAE-Results/256-0.75/results/F2/models/{mask_type}_0.07_{mask_type}_0.04/{mask_type}/best_model_lf_0.03.pth",
    #  f"./interp/{sparse_type}/F2"),
]
for model_path, df_path in model_paths: 
    seed = 42
    generator = torch.Generator().manual_seed(seed)

    NUM_WORKERS = 4
    if device.type.lower() == "cpu":
        NUM_WORKERS = 0

    # training data
    train_dataset = EdgeDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker, generator=generator, pin_memory=True)

    # test data
    test_dataset = EdgeDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = NeuralNetwork().to(device)
    sae_hidden_one = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)
    sae_hidden_two = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    sae_hidden_one.load_state_dict(checkpoint['sae_one_state_dict'])
    sae_hidden_two.load_state_dict(checkpoint['sae_two_state_dict'])

    train_results = evaluate_and_gather_activations(model, sae_hidden_one, sae_hidden_two, train_loader, device)
    Z_train_one, Z_train_two, y_train = train_results["Z_one"], train_results["Z_two"], train_results["y"]

    test_results = evaluate_and_gather_activations(model, sae_hidden_one, sae_hidden_two, test_loader, device)
    Z_test_one, Z_test_two, y_test = test_results["Z_one"], test_results["Z_two"], test_results["y"]

    print(f"Model acc: {test_results['accuracy']}")

    activation_data = extract_activations(
        data_loader=test_loader,
        model=model,
        sae_one=sae_hidden_one,
        sae_two=sae_hidden_two,
        device=device
    )

    sparse_act_one = activation_data["sparse_one"]
    max_feature_indices_sp_one = np.argmax(sparse_act_one, axis=1)


    # Suppose:
    X = activation_data['sparse_one']        # your n×d sparse feature matrix
    true_labels = activation_data['labels']  # ground-truth digits
    pred_labels = max_feature_indices_sp_one  # argmax of your sparse codes

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

    os.makedirs(df_path, exist_ok=True)
    metrics_df['Value'] = metrics_df['Value'].round(3)
    metrics_df.to_csv(f"{df_path}/metrics.csv")
    print(metrics_df)