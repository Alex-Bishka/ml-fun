import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


from models_and_data.nn import NeuralNetwork
from models_and_data.edgedataset import EdgeDataset
from models_and_data.model_helpers import load_intermediate_labels, seed_worker, set_seed


SEED = 42
NUM_EPOCHS = 100
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

print("Train images shape:", train_images.shape)
print("Val images shape:", val_images.shape)
print("Test images shape:", test_images.shape)

NUM_WORKERS = 4
if device.type.lower() == "cpu":
    NUM_WORKERS = 0
generator = torch.Generator().manual_seed(SEED)

# training data
train_dataset = EdgeDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=generator)

# validation data
val_dataset = EdgeDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)  # larger batch size for faster validation

# test data
test_dataset = EdgeDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)


set_seed(SEED)
model = NeuralNetwork().to(device)
classification_loss_fn = nn.CrossEntropyLoss()
hidden_act_one_loss_fn = nn.CosineSimilarity()
optimizer = torch.optim.Adam(model.parameters())
print(f"Model weights (to compare below): {model.hidden_one.weight[0][:5].detach().cpu().numpy()}")


best_model = None
best_val_acc = 0.0
best_val_loss = float('inf')

validation_losses = []
training_losses = []

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()  # set the model to training mode - this is currently a no-op
    
    train_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]", leave=False)
    for batch_idx, batch in enumerate(train_bar):
        # deconstruct batch items
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        classification_out, hidden_act_one, hidden_act_two = model(images)

        # Classification loss and backprop
        optimizer.zero_grad()
        total_loss = classification_loss_fn(classification_out, labels)
        total_loss.backward()
        
        optimizer.step()
        train_loss += total_loss.item()
        train_bar.set_postfix(loss=total_loss.item())

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            # deconstruct
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # forward pass
            classification_out, _, _ = model(images)

            # compute loss
            loss = classification_loss_fn(classification_out, labels)

            # calculate metrics
            val_loss += loss.item()
            _, predicted = torch.max(classification_out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # epoch stats
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        
        model_path = f'./total_epoch_{NUM_EPOCHS}/model_{epoch+1}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
        torch.save({
            'model_state_dict': model.state_dict()
        }, model_path)
        best_model_path = model_path
        print(f"  -> Saved best model checkpoint to {best_model_path}")

    validation_losses.append(avg_val_loss)
    training_losses.append(avg_train_loss)


print(f"Best model path choosen: '{best_model_path}'")
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

total_correct = 0
total_samples = 0

with torch.no_grad():
    test_bar = tqdm(test_loader, desc=f"Evaluation")
    for i, batch in enumerate(test_bar):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # forward pass
        classification_out, hidden_one_act, hidden_two_act = model(images)

        # stats
        _, predicted = torch.max(classification_out, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

accuracy = 100 * total_correct / total_samples
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")