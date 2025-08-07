import numpy as np
from sklearn.linear_model import LogisticRegression

import torch
from torch import nn 
from torch.utils.data import DataLoader

from models_and_data.nn import NeuralNetwork
from models_and_data.sae import SparseAutoencoder
from models_and_data.edgedataset import EdgeDataset

from models_and_data.model_helpers import (evaluate_and_gather_activations, load_intermediate_labels, seed_worker)


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

train_dataset = EdgeDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS,
                          worker_init_fn=seed_worker, generator=generator, pin_memory=True)
# test data
test_dataset = EdgeDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model = NeuralNetwork().to(device)
sae_hidden_one = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)
sae_hidden_two = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)

best_model_path = "./full-total_epoch_100/full-70.pth"
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
sae_hidden_one.load_state_dict(checkpoint['sae_one_state_dict'])
sae_hidden_two.load_state_dict(checkpoint['sae_two_state_dict'])

train_results = evaluate_and_gather_activations(model, sae_hidden_one, sae_hidden_two, train_loader, device)
Z_train_one, Z_train_two, y_train = train_results["Z_one"], train_results["Z_two"], train_results["y"]

test_results = evaluate_and_gather_activations(model, sae_hidden_one, sae_hidden_two, test_loader, device)
Z_test_one, Z_test_two, y_test = test_results["Z_one"], test_results["Z_two"], test_results["y"]

print(f"Model acc: {test_results['accuracy']}")

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