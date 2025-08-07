import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression

from models_and_data.sae import SparseAutoencoder
from models_and_data.nn import NeuralNetwork
from models_and_data.edgedataset import EdgeDataset
from models_and_data.model_helpers import (load_intermediate_labels, seed_worker, set_seed, evaluate_and_gather_activations)


HIDDEN_SIZE = 256
L1_PENALTY = 0.75
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"We will be using device: {device}")

# train data
train_images = load_intermediate_labels("./intermediate-labels/first_layer/train_images.pkl")
train_labels = load_intermediate_labels("./intermediate-labels/first_layer/train_labels.pkl")

# val data
val_images = load_intermediate_labels("./intermediate-labels/first_layer/val_images.pkl")
val_labels = load_intermediate_labels("./intermediate-labels/first_layer/val_labels.pkl")

# test data
test_images = load_intermediate_labels("./intermediate-labels/first_layer/test_images.pkl")
test_labels = load_intermediate_labels("./intermediate-labels/first_layer/test_labels.pkl")

seed = 42
generator = torch.Generator().manual_seed(seed)

NUM_WORKERS = 4
if device.type.lower() == "cpu":
    NUM_WORKERS = 0

# training data
train_dataset = EdgeDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS,
                        worker_init_fn=seed_worker, generator=generator, pin_memory=True)

# validation data
val_dataset = EdgeDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)  # larger batch size for faster validation
            
# test data
test_dataset = EdgeDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# best_model_path = "./SAE-Results/256-0.75/results/baseline/model_state_dict.pth"
# best_model_path = "./SAE-Results/256-0.75/results/F2/models/25_top_0.14_25_top_0.06/25_top/best_model_lf_0.18.pth"
# checkpoint = torch.load(best_model_path)

set_seed(42)
model = NeuralNetwork().to(device)
# model.load_state_dict(checkpoint['model_state_dict'])


set_seed(42)
sae_hidden_one = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)

set_seed(42)
sae_hidden_two = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)

classification_loss_fn = nn.CrossEntropyLoss()
hidden_act_one_loss_fn = nn.CosineSimilarity()

optimizer = torch.optim.Adam(model.parameters())
optimizer_sae_hidden_one = torch.optim.Adam(sae_hidden_one.parameters())
optimizer_sae_hidden_two = torch.optim.Adam(sae_hidden_two.parameters())


num_epochs = 100    
best_val_acc = 0.0
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()  # set the model to training mode - this is currently a no-op
    sae_hidden_two.train()
    sae_hidden_one.train()
    
    train_loss, sae_loss_one, sae_loss_two = 0.0, 0.0, 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
    for batch_idx, batch in enumerate(train_bar):
        # deconstruct batch items
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        optimizer.zero_grad()
        classification_out, hidden_act_one, hidden_act_two = model(images)            
        total_loss = classification_loss_fn(classification_out, labels) 
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        train_bar.set_postfix(loss=total_loss.item())

        # to prevent backprop on both graphs:
        hidden_act_one_detached = hidden_act_one.detach()
        hidden_act_two_detached = hidden_act_two.detach()

        # SAE loss and backprop - hidden layer one
        optimizer_sae_hidden_one.zero_grad()
        reconstructed_one, encoded_one = sae_hidden_one(hidden_act_one_detached)
        loss1 = sae_hidden_one.loss(hidden_act_one_detached,
                                                reconstructed_one,
                                                encoded_one,
                                                l1_lambda=L1_PENALTY
                                                )
        loss1.backward()
        optimizer_sae_hidden_one.step()
        sae_loss_one += loss1.item()
        
        # SAE loss and backprop - hidden layer two
        optimizer_sae_hidden_two.zero_grad()
        reconstructed_two, encoded_two = sae_hidden_two(hidden_act_two_detached)
        loss2 = sae_hidden_two.loss(hidden_act_two_detached,
                                                reconstructed_two,
                                                encoded_two,
                                                l1_lambda=L1_PENALTY
                                                )
        loss2.backward()
        optimizer_sae_hidden_two.step()
        sae_loss_two += loss2.item()

    # --- Validation Phase ---
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            # deconstruct
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            classification_out, _, _ = model(images)
            loss = classification_loss_fn(classification_out, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(classification_out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # epoch stats
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        
        model_path = f'./full-total_epoch_{num_epochs}/full-{epoch}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Save all three model states in one file
        torch.save({
            'model_state_dict': model.state_dict(),
            'sae_one_state_dict': sae_hidden_one.state_dict(),
            'sae_two_state_dict': sae_hidden_two.state_dict(),
        }, model_path)
        best_model_path = model_path
        print(f"  -> Saved best model checkpoint to {best_model_path}")


