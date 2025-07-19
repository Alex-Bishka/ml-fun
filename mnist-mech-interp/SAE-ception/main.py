# Saving this for later
# probably the closest settings we have to mimicing
# the 99.5% accuracy on ViT-H

import torch
from torchvision import datasets, transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import interpolate

from helpers.helpers import set_seed

# --- 0. For Reproducibility ---
set_seed(42)
BATCH_SIZE = 8
IMG_RES = 384
GRAD_ACCUM_STEPS = 64  # Optional: Accumulate gradients over 2 steps to reduce memory
TOTAL_STEPS = 10000
BASE_LR = 0.01

# --- 1. Model and Device Setup ---
weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
model = vit_h_14(weights=weights)


# Update model's image size and positional embeddings
model.image_size = IMG_RES  # Update the expected image size
patch_size = model.patch_size  # Should be 14 for vit_h_14
num_patches = (IMG_RES // patch_size) ** 2  # 729 for 384x384 with 14x14 patches

# Interpolate positional embeddings
orig_pos_embed = model.encoder.pos_embedding  # Shape: [1, 257, 1280] (257 = 1 cls token + 256 patches)
new_num_patches = num_patches + 1  # +1 for the class token
grid_size = int(num_patches ** 0.5)  # 27 for 729 patches

# Extract the positional embeddings (excluding class token)
pos_embed = orig_pos_embed[:, 1:, :]  # Shape: [1, 256, 1280]
pos_embed = pos_embed.reshape(1, 16, 16, -1)  # Reshape to [1, 16, 16, 1280] (for 224x224)
pos_embed = torch.nn.functional.interpolate(
    pos_embed.permute(0, 3, 1, 2),  # [1, 1280, 16, 16]
    size=(grid_size, grid_size),  # Interpolate to [1, 1280, 27, 27]
    mode='bilinear',
    align_corners=False
)
pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, -1)  # [1, 729, 1280]

# Combine with class token embedding
cls_token_embed = orig_pos_embed[:, :1, :]  # [1, 1, 1280]
new_pos_embed = torch.cat([cls_token_embed, pos_embed], dim=1)  # [1, 730, 1280]

# Update model's positional embeddings
model.encoder.pos_embedding = torch.nn.Parameter(new_pos_embed)


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
eval_transform = transforms.Compose([
    transforms.Resize((IMG_RES, IMG_RES)),  # Ensure 384x384 for validation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
val_dataset_for_split = datasets.CIFAR10(root='./data', train=True, download=True, transform=eval_transform)

num_train = len(full_train_dataset)
split = int(0.9 * num_train)
indices = torch.randperm(num_train).tolist()

train_subset = torch.utils.data.Subset(full_train_dataset, indices[:split])
val_subset = torch.utils.data.Subset(val_dataset_for_split, indices[split:])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
NUM_EPOCHS = TOTAL_STEPS // steps_per_epoch
print(f"Dataset has {len(train_loader)} batches per epoch.")
print(f"Training for {TOTAL_STEPS} steps, which is ~{NUM_EPOCHS} epochs.")

# --- 3. Optimizer, Loss Function, and GradScaler ---
optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=0)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')
scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=0)

# --- 4. The Fine-Tuning and Validation Loop ---
best_val_accuracy = 0.0
MODEL_SAVE_PATH = 'temp.pth'

print("🚀 Starting fine-tuning...")
for epoch in range(NUM_EPOCHS):
    # -- Training Phase --
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Mixed precision context
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * GRAD_ACCUM_STEPS
        if (i + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # -- Validation Phase --
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):  # Mixed precision for validation
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