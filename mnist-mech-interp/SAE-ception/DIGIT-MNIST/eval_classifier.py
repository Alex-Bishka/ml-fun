from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


from models_and_data.nn import NeuralNetwork
from models_and_data.edgedataset import EdgeDataset
from models_and_data.model_helpers import (load_intermediate_labels, seed_worker)


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

model = NeuralNetwork().to(device)
# best_model_path = "./SAE-Results/256-0.75/results/baseline/model_state_dict.pth"
best_model_path = "./SAE-Results/256-0.75/results/F2/models/25_top_0.14_25_top_0.06/25_top/best_model_lf_0.18.pth"
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])


model.eval()
total_correct = 0
total_samples = 0
target_labels = [0]
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating and Gathering Activations", leave=False):
        if len(batch) == 3:
            images, labels, _ = batch # Unpack but ignore the third item
        else:
            images, labels = batch

        images, labels = images.to(device), labels.to(device)

        mask = torch.isin(labels, torch.tensor(target_labels, device=device))
        if mask.sum() > 0:  # Only process if there are samples with target labels
            images_filtered = images[mask]
            labels_filtered = labels[mask]

            # --- Main Model Forward Pass ---
            classification_out, _, _ = model(images_filtered)
            
            _, predicted = torch.max(classification_out, 1)
            total_correct += (predicted == labels_filtered).sum().item()
            total_samples += labels_filtered.size(0)

accuracy = 100 * total_correct / total_samples
print(f"Accuracy of model: {accuracy}")