import torch
from torchvision import datasets, transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torch.utils.data import DataLoader
from torchvision.transforms import autoaugment
from timm.data.mixup import Mixup


from helpers.sae import SparseAutoencoder
from helpers.helpers import set_seed, seed_worker


#  --- 0. For reproducibility and setup ---
BATCH_SIZE = 2
ACCUMULATION_STEPS = 32
NUM_WORKERS = 4
set_seed(42)

captured_activations = {}
def get_activation(name):
    """
    This function returns another function (a closure) that will be our hook.
    The hook function saves the input of a module in our dictionary.
    """
    def hook(model, input, output):
        # The input to the MLP is a tuple, we want the tensor inside.
        captured_activations[name] = input[0].detach()
    return hook

# --- 1. Model and Device Setup ---
# weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1    # for 224 resolution
weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1       # for 518 resolution
model = vit_h_14(weights=weights)

num_ftrs = model.heads.head.in_features
model.heads.head = torch.nn.Linear(num_ftrs, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Attaching the hooks ---
vit_h_14_num_layers = 32
first_layer_index = 0
middle_layer_index = vit_h_14_num_layers // 2 - 1
last_layer_index = vit_h_14_num_layers - 1

model.encoder.layers[first_layer_index].mlp.register_forward_hook(get_activation('first_mlp'))
model.encoder.layers[middle_layer_index].mlp.register_forward_hook(get_activation('middle_mlp'))
model.encoder.layers[last_layer_index].mlp.register_forward_hook(get_activation('last_mlp'))

# --- 2. Data Preparation and Splitting ---
train_transform = transforms.Compose([
    transforms.Resize((518, 518)),
    autoaugment.RandAugment(num_ops=9, magnitude=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Resize((518, 518)), # Change to 384x384
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

generator = torch.Generator().manual_seed(42)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                         worker_init_fn=seed_worker, generator=generator,
                         num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                         generator=generator, num_workers=NUM_WORKERS,
                         pin_memory=True)


# --- 3. Optimizer and Loss Function ---
num_epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_epochs)


# --- 4. The Fine-Tuning and Validation Loop ---
best_val_accuracy = 0.0
MODEL_SAVE_PATH = 'vit_h_14_cifar10_best.pth'
mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, num_classes=10)

mlp_input_dim = 1280

sae_first = SparseAutoencoder(input_dim=mlp_input_dim).to(device)
sae_middle = SparseAutoencoder(input_dim=mlp_input_dim).to(device)
sae_last = SparseAutoencoder(input_dim=mlp_input_dim).to(device)

optimizer_sae_first = torch.optim.Adam(sae_first.parameters())
optimizer_sae_middle = torch.optim.Adam(sae_middle.parameters())
optimizer_sae_last = torch.optim.Adam(sae_last.parameters())

print("🚀 Starting fine-tuning...")
for epoch in range(num_epochs):
    # -- Training Phase --
    model.train()
    sae_first.train()
    sae_middle.train()
    sae_last.train()

    running_loss, running_sae_first_loss, running_sae_middle_loss, running_sae_last_loss = 0.0, 0.0, 0.0, 0.0

    optimizer.zero_grad()
    optimizer_sae_first.zero_grad()
    optimizer_sae_middle.zero_grad()
    optimizer_sae_last.zero_grad()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images, labels = mixup_fn(images, labels)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        act_first = captured_activations['first_mlp']
        act_middle = captured_activations['middle_mlp']
        act_last = captured_activations['last_mlp']

        recon_first, encoded_first = sae_first(act_first)
        loss_first = sae_first.loss(act_first, recon_first, encoded_first) / ACCUMULATION_STEPS
        loss_first.backward()

        recon_middle, encoded_middle = sae_middle(act_middle)
        loss_middle = sae_middle.loss(act_middle, recon_middle, encoded_middle) / ACCUMULATION_STEPS
        loss_middle.backward()

        recon_last, encoded_last = sae_last(act_last)
        loss_last = sae_last.loss(act_last, recon_last, encoded_last) / ACCUMULATION_STEPS
        loss_last.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            optimizer_sae_first.step()
            optimizer_sae_first.zero_grad()

            optimizer_sae_middle.step()
            optimizer_sae_middle.zero_grad()

            optimizer_sae_last.step()
            optimizer_sae_last.zero_grad()

        running_loss += loss.item() * ACCUMULATION_STEPS
        running_sae_first_loss += loss_first.item() * ACCUMULATION_STEPS
        running_sae_middle_loss += loss_middle.item() * ACCUMULATION_STEPS
        running_sae_last_loss += loss_last.item() * ACCUMULATION_STEPS
        if (i + 1) % 100 == 0:
            avg_main_loss = running_loss / 100
            avg_sae_first = running_sae_first_loss / 100
            avg_sae_middle = running_sae_middle_loss / 100
            avg_sae_last = running_sae_last_loss / 100

            print(
                f"  Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
                f"Loss: {avg_main_loss:.4f}, "
                f"SAE First Loss: {avg_sae_first:.4f}, "
                f"SAE Middle Loss: {avg_sae_middle:.4f}, "
                f"SAE Last Loss: {avg_sae_last:.4f}"
            )

            running_loss = 0.0
            running_sae_first_loss = 0.0
            running_sae_middle_loss = 0.0
            running_sae_last_loss = 0.0

    # -- Validation Phase --
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

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
    print(f"SAE first loss: {avg_sae_first:.4f}, SAE middle loss: {avg_sae_middle:.4f}, SAE last loss: {avg_sae_middle:.4f}")

    # -- Save the Best Model --
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            'sae_first_state_dict': sae_first.state_dict(),
            'sae_middle_state_dict': sae_middle.state_dict(),
            'sae_last_state_dict': sae_last.state_dict(),
        }, MODEL_SAVE_PATH)
        print(f"✨ New best model saved with accuracy: {val_accuracy:.2f}% to {MODEL_SAVE_PATH}")

    scheduler.step()

print("✅ Fine-tuning finished.")


# --- 5. Final Evaluation on Test Set ---
print("\n🧪 Starting final evaluation on the test set...")

# Load the best performing model
checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Create the test DataLoader
generator = torch.Generator().manual_seed(42)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                         generator=generator, num_workers=NUM_WORKERS,
                         pin_memory=True)

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