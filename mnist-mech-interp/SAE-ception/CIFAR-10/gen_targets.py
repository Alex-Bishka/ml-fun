import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

import torch
from torch.utils.data import DataLoader


from helpers.nn import NeuralNetwork
from helpers.sae import SparseAutoencoder
from helpers.edgedataset import EdgeDataset

from helpers.model_helpers import (evaluate_and_gather_activations, get_sublabel_data, 
    get_top_N_features, extract_activations, load_intermediate_labels, seed_worker)


# Parameters
HIDDEN_SIZE = 1024
INPUT_SIZE = 64

# F0 features
best_model_path = "./runs/1024-0.75/results/F0/models/256_top/best_model_lf_0.19.pth"

# baseline
# best_model_path = "./runs/1024-0.75/results/baseline/model_state_dict.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"We will be using device: {device}")

# train data
train_images = load_intermediate_labels("./data/FashionMNIST/parsed/train_images.pkl")
train_labels = load_intermediate_labels("./data/FashionMNIST/parsed/train_labels.pkl")

# test data
test_images = load_intermediate_labels("./data/FashionMNIST/parsed/test_images.pkl")
test_labels = load_intermediate_labels("./data/FashionMNIST/parsed/test_labels.pkl")

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

# init and load model
model = NeuralNetwork().to(device)
sae_hidden_one = SparseAutoencoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
sae_hidden_two = SparseAutoencoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
sae_hidden_three = SparseAutoencoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)

checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
sae_hidden_one.load_state_dict(checkpoint['sae_one_state_dict'])
sae_hidden_two.load_state_dict(checkpoint['sae_two_state_dict'])
sae_hidden_three.load_state_dict(checkpoint['sae_three_state_dict'])

# get evaluation results
train_results = evaluate_and_gather_activations(model, sae_hidden_one, sae_hidden_two, sae_hidden_three, train_loader, device)
Z_train_one, Z_train_two, y_train = train_results["Z_one"], train_results["Z_two"], train_results["y"]
test_results = evaluate_and_gather_activations(model, sae_hidden_one, sae_hidden_two, sae_hidden_three, test_loader, device)
Z_test_one, Z_test_two, y_test = test_results["Z_one"], test_results["Z_two"], test_results["y"]
print(f"Model acc: {test_results['accuracy']}")

# evaluate SAEs
sparsity_one = np.mean(Z_test_one > 1e-5) * Z_test_one.shape[1]
sparsity_two = np.mean(Z_test_two > 1e-5) * Z_test_two.shape[1]
print(f"Average Non-Zero Features per Image (Hidden One): {sparsity_one:.2f}")
print(f"Average Non-Zero Features per Image (Hidden Two): {sparsity_two:.2f}")

print("\n--- Training Linear Probes ---")
clf_one = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
clf_one.fit(Z_train_one, y_train)
acc_one = clf_one.score(Z_test_one, y_test)
print(f"Linear Probe Accuracy (Hidden One): {acc_one:.2%}")

clf_two = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
clf_two.fit(Z_train_two, y_train)
acc_two = clf_two.score(Z_test_two, y_test)
print(f"Linear Probe Accuracy (Hidden Two): {acc_two:.2%}")

# Target recon generation
activation_data = extract_activations(
    data_loader=train_loader,
    model=model,
    sae_one=sae_hidden_one,
    sae_two=sae_hidden_two,
    device=device
)

sparse_vector_sizes = [25, 256]
for N_recon in sparse_vector_sizes:
    labels = activation_data["labels"]
    sparse_act_one = activation_data["sparse_one"]
    avg_digit_encoding, top_n_features = get_top_N_features(N_recon, sparse_act_one, labels)
    
    feature_indices_dict = {}
    for digit in range(0, 10):
        feature_indices_dict[digit] = top_n_features[digit]['indices']
    
    print("Features used:")
    print(len(feature_indices_dict[0]))
    
    recon_max_sparse_training, recon_max_sparse_ablated_training = get_sublabel_data(
                                                                    train_labels,
                                                                    train_images,
                                                                    feature_indices_dict,
                                                                    sparse_act_one,
                                                                    sae_hidden_one,
                                                                    device,
                                                                    HIDDEN_SIZE
                                                                )
    
    print("Size of datasets:")
    print(len(train_images), len(test_images), len(recon_max_sparse_training))
    
    file_path = f"./{N_recon}_top.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(recon_max_sparse_training, f)
    
    file_path = f"./{N_recon}_mask.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(recon_max_sparse_ablated_training, f)