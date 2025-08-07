import pandas as pd
import torch
from torchvision import datasets, transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torch.utils.data import DataLoader


from helpers.helpers import set_seed
from helpers.sae import SparseAutoencoder, train_sae_on_layer, evaluate_sae_with_probe


# --- 0. For reproducibility & Configuration ---
set_seed(42)
BATCH_SIZE = 64
SAE_EPOCHS = 5
# MODEL_SAVE_PATH = './classifiers/baseline/vit_h_99.56.pth'
# MODEL_SAVE_PATH = './classifiers/F0/vit_h_99.56_25_top_0.0002_99.41.pth'
MODEL_SAVE_PATH = './classifiers/F1/best_model_lf_0.01.pth'  # 99.30% accuracy on this guy
SAE_BASE_PATH = './sae_models/F1'
IMG_RES = 384
FEATURE_DIM = 1280

print("\n--- 1. Loading ViT Model ---")
weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
model = vit_h_14(weights=weights) 


# Update model's image size and positional embeddings
model.image_size = IMG_RES  # Update the expected image size
patch_size = model.patch_size  # Should be 14 for vit_h_14
num_patches = (IMG_RES // patch_size) ** 2  # 729 for 384x384 with 14x14 patches

# Interpolate positional embeddings
orig_pos_embed = model.encoder.pos_embedding  # Shape: [1, 257, 1280] (257 = 1 cls token + 256 patches)
print(f"Original pos_embed shape: {orig_pos_embed.shape}")

# Extract the embedding dimension
embed_dim = orig_pos_embed.shape[-1]  # 1280
num_orig_patches = orig_pos_embed.shape[1] - 1  # 1369 patches (exclude class token)
orig_grid_size = int(num_orig_patches ** 0.5)  # 37 for 1369 patches (37x37 grid)
new_grid_size = int(num_patches ** 0.5)  # 27 for 729 patches (27x27 grid)

# Extract the positional embeddings (excluding class token)
pos_embed = orig_pos_embed[:, 1:, :]  # Shape: [1, 1369, 1280]
pos_embed = pos_embed.reshape(1, orig_grid_size, orig_grid_size, embed_dim)  # Reshape to [1, 37, 37, 1280]

# Interpolate to new grid size
pos_embed = torch.nn.functional.interpolate(
    pos_embed.permute(0, 3, 1, 2),  # [1, 1280, 37, 37]
    size=(new_grid_size, new_grid_size),  # Interpolate to [1, 1280, 27, 27]
    mode='bilinear',
    align_corners=False
)
pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, embed_dim)  # [1, 729, 1280]

# Combine with class token embedding
cls_token_embed = orig_pos_embed[:, :1, :]  # [1, 1, 1280]
new_pos_embed = torch.cat([cls_token_embed, pos_embed], dim=1)  # [1, 730, 1280]

# Update model's positional embeddings
model.encoder.pos_embedding = torch.nn.Parameter(new_pos_embed)


num_ftrs = model.heads.head.in_features
model.heads.head = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Successfully loaded model from {MODEL_SAVE_PATH} to device: {device}")


print("\n--- 2. Preparing Data for SAE Training and Validation ---")
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

print(f"SAE training set size: {len(train_subset)}")
print(f"SAE validation set size: {len(val_subset)}")


print("\n--- 3. Starting Decoupled SAE Training ---")
target_layers_config = {
    # "first_layer": {
    #     "layer": model.encoder.layers[0],
    #     "dim": FEATURE_DIM,
    #     "L1": 5e-7
    # },
    # "middle_layer": {
    #     "layer": model.encoder.layers[15],
    #     "dim": FEATURE_DIM,
    #     "L1": 5e-5
    # },
    "last_layer": {
        "layer": model.encoder,  # potentially want to also look at 'model.encoder.layers[-1]'
        "dim": FEATURE_DIM,
        "L1": 2e-4
    }
}

L1_config = {
    # "first_layer":  [5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9],
    # "middle_layer": [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8],
    # "last_layer":   [5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
    "last_layer":   [2e-4]
}

results_df = pd.DataFrame(columns=['layer_name', 'l1_penalty', 'accuracy', 'sparsity'])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
for layer_name, config in target_layers_config.items():
    for l1_penalty in L1_config[layer_name]:
        print(f"Using L1 penalty '{l1_penalty}' for layer '{layer_name}'")
        sae_save_path = f"{SAE_BASE_PATH}/sae_{layer_name}_l1_{l1_penalty}.pth"

        train_sae_on_layer(
            vit_model=model,
            target_layer=config["layer"],
            layer_name=layer_name,
            sae_input_dim=config["dim"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            sae_epochs=SAE_EPOCHS,
            sae_l1_lambda=l1_penalty,
            sae_save_path=sae_save_path
            # sae_l1_lambda=config["L1"]
        )

        sae = SparseAutoencoder(input_dim=config["dim"]).to(device)
        sae.load_state_dict(torch.load(sae_save_path))

        accuracy, sparsity = evaluate_sae_with_probe(
            vit_model=model,
            sae_model=sae,
            target_layer=config["layer"],
            layer_name=layer_name,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device
        )

        new_row = pd.DataFrame([{
            'layer_name': layer_name,
            'l1_penalty': l1_penalty,
            'accuracy': accuracy,
            'sparsity': sparsity
        }])

        results_df = pd.concat([results_df, new_row], ignore_index=True)

results_df.to_csv("sae_l1_experiment_results.csv", index=False)
print("\n🎉 All SAEs trained successfully.")

