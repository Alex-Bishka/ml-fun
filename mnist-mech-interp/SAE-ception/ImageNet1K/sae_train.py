import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from datasets import load_dataset

from helpers.helpers import set_seed
from helpers.sae import SparseAutoencoder, train_sae_on_layer, evaluate_sae_with_probe


# --- 0. For reproducibility & Configuration ---
SEED = 42
MODEL_LOAD_PATH = './SAE-Results/results/baseline/baseline_classifier.pth'  # convnextv2_large.fcmae_ft_in22k_in1k_384_baseline.pth
SAE_SAVE_PATH = './sae_models/baseline'

BATCH_SIZE = 128
SAE_EPOCHS = 5

VAL_SET_SIZE = 25000
IMG_RES = 384
FEATURE_DIM = 1536  # dimensions of targeted layer: Flatten: 2-9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)


# --- Loading Convnext Model ---
print("\n--- 1. Convnext Model ---")
model = timm.create_model(
    'convnextv2_large.fcmae_ft_in22k_in1k_384',
    pretrained=False,
    num_classes=1000  # ImageNet-1K
)
model = torch.load(MODEL_LOAD_PATH)
model.to(device)

print(f"Successfully loaded model from {MODEL_LOAD_PATH} to device: {device}")


# --- Preparing dataset ---
print("\n--- 2. Preparing Data for SAE Training and Validation ---")
full_train_split = load_dataset("ILSVRC/imagenet-1k", split='train', cache_dir='./data')
shuffled_train = full_train_split.shuffle(seed=SEED, buffer_size=50000)

val_split = shuffled_train.take(VAL_SET_SIZE)
train_split = shuffled_train.skip(VAL_SET_SIZE)
test_split = load_dataset("ILSVRC/imagenet-1k", split='validation', cache_dir='./data')     # standard to use val as test for Image Net 1k

tfms = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=model.default_cfg['mean'],
                         std=model.default_cfg['std'])
])

def collate(batch):
    imgs = torch.stack([tfms(x['image']) for x in batch])
    labels = torch.tensor([x['label'] for x in batch])
    return imgs, labels

train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, collate_fn=collate,
                          num_workers=train_split.n_shards, pin_memory=True)
val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, collate_fn=collate, 
                        num_workers=val_split.n_shards, pin_memory=True)
test_loader = DataLoader(test_split, batch_size=BATCH_SIZE, collate_fn=collate,
                         num_workers=test_split.n_shards, pin_memory=True)


# --- Decoupled Training ---
print("\n--- 3. Starting Decoupled SAE Training ---")
target_layers_config = {
    "last_layer": {
        "layer": model.head.flatten,  # potentially want to also look at 'model.encoder.layers[-1]'
        "dim": FEATURE_DIM
    }
}

L1_config = {
    # "last_layer":   [5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
    "last_layer":   [5e-4, 2e-4, 1e-5]
}

results_df = pd.DataFrame(columns=['layer_name', 'l1_penalty', 'accuracy', 'sparsity'])
for layer_name, config in target_layers_config.items():
    for l1_penalty in L1_config[layer_name]:
        print(f"Using L1 penalty '{l1_penalty}' for layer '{layer_name}'")
        sae_save_path = f"{SAE_SAVE_PATH}/sae_{layer_name}_l1_{l1_penalty}.pth"
        os.makedirs(SAE_SAVE_PATH, exist_ok=True)

        train_sae_on_layer(
            model=model,
            target_layer=config["layer"],
            layer_name=layer_name,
            sae_input_dim=config["dim"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            sae_epochs=SAE_EPOCHS,
            sae_l1_lambda=l1_penalty,
            sae_save_path=sae_save_path,
            val_set_size=VAL_SET_SIZE
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