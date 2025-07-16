import torch
from torchvision import datasets, transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torch.utils.data import DataLoader


from helpers.helpers import set_seed


#  --- 0. For reproducibility ---
set_seed(42)
BATCH_SIZE = 16
ACCUMULATION_STEPS = 32
IMG_RES = 224
NUM_EPOCHS = 15


# --- 1. Model and Device Setup ---
weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1  # Using LINEAR for efficiency
model = vit_h_14(weights=weights)

num_ftrs = model.heads.head.in_features
model.heads.head = torch.nn.Linear(num_ftrs, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# --- 2. Data Preparation and Splitting ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_RES, IMG_RES)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_transform = weights.transforms()

full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
val_dataset_for_split = datasets.CIFAR10(root='./data', train=True, download=True, transform=eval_transform)

num_train = len(full_train_dataset)
split = int(0.9 * num_train)
indices = torch.randperm(num_train).tolist()

train_subset = torch.utils.data.Subset(full_train_dataset, indices[:split])
val_subset = torch.utils.data.Subset(val_dataset_for_split, indices[split:])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)


# --- 3. Optimizer and Loss Function ---
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()


# --- 4. The Fine-Tuning and Validation Loop ---
best_val_accuracy = 0.0
MODEL_SAVE_PATH = 'vit_h_14_cifar10_best.pth'

print("🚀 Starting fine-tuning...")
for epoch in range(NUM_EPOCHS):
    # -- Training Phase --
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / ACCUMULATION_STEPS
        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * ACCUMULATION_STEPS
        if (i + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / (100 / ACCUMULATION_STEPS):.4f}")
            running_loss = 0.0

    # -- Validation Phase --
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Disable gradient calculation for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"✅ End of Epoch {epoch+1}: Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # -- Save the Best Model --
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✨ New best model saved with accuracy: {val_accuracy:.2f}% to {MODEL_SAVE_PATH}")

print("✅ Fine-tuning finished.")


# --- 5. Final Evaluation on Test Set ---
print("\n🧪 Starting final evaluation on the test set...")

# Load the best performing model
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(device)
model.eval()

# Create the test DataLoader
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate the model
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

final_accuracy = 100 * test_correct / test_total
print(f"\n🎉 Final Accuracy of the best model on the test set: {final_accuracy:.2f}%")