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
MODEL_SAVE_PATH = './classifiers/baseline/vit_h_99.02.pth'
SAE_BASE_PATH = './sae_models/baseline'
IMG_RES = 224
FEATURE_DIM = 1280

print("\n--- 1. Loading ViT Model ---")
weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
model = vit_h_14(weights=None) 
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

print(f"SAE training set size: {len(train_subset)}")
print(f"SAE validation set size: {len(val_subset)}")


print("\n--- 3. Starting Decoupled SAE Training ---")
target_layers_config = {
    "first_layer": {
        "layer": model.encoder.layers[0],
        "dim": FEATURE_DIM,
        "L1": 5e-7
    },
    "middle_layer": {
        "layer": model.encoder.layers[15],
        "dim": FEATURE_DIM,
        "L1": 5e-5
    },
    "last_layer": {
        "layer": model.encoder,  # potentially want to also look at 'model.encoder.layers[-1]'
        "dim": FEATURE_DIM,
        "L1": 2e-4
    }
}

L1_config = {
    "first_layer":  [5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9],
    "middle_layer": [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8],
    "last_layer":   [5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
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
