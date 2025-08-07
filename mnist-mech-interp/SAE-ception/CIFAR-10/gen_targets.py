import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pickle
from torchvision import datasets, transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torch.utils.data import DataLoader


from helpers.sae import SparseAutoencoder
from helpers.helpers import set_seed, extract_activations, get_top_N_features, get_sublabel_data


# --- 0. For reproducibility & Configuration ---
set_seed(42)
BATCH_SIZE = 64
SAE_EPOCHS = 5
# MODEL_LOAD_PATH = './classifiers/baseline/vit_h_99.56.pth'
MODEL_LOAD_PATH = './classifiers/F0/vit_h_99.56_25_top_0.0002_99.41.pth'
# SAE_MODEL_PATH = "./sae_models/baseline-99.56/last_layer/sae_last_layer_l1_0.0002.pth"
SAE_MODEL_PATH = "./sae_models/F0/sae_last_layer_l1_0.0002.pth"
IMG_RES = 384
FEATURE_DIM = 1280

N_TOP_FEATURES = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n--- 1. Loading ViT Model and SAE Model ---")
weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
model = vit_h_14(weights=weights) 
model.image_size = IMG_RES  # Update the expected image size

model.encoder.pos_embedding = torch.nn.Parameter(torch.load('./embeds/pos_embed_edge_384_99.56.pth'))

num_ftrs = model.heads.head.in_features
model.heads.head = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(MODEL_LOAD_PATH))
model.to(device)

sae = SparseAutoencoder(input_dim=FEATURE_DIM).to(device)
sae.load_state_dict(torch.load(SAE_MODEL_PATH))


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

seed = 42
generator = torch.Generator().manual_seed(seed)
indices = torch.randperm(num_train, generator=generator).tolist()

train_subset = torch.utils.data.Subset(full_train_dataset, indices[:split])
val_subset = torch.utils.data.Subset(val_dataset_for_split, indices[split:])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"SAE training set size: {len(train_subset)}")
print(f"SAE validation set size: {len(val_subset)}")

print("\n--- 3. Extracting Activations ---")
activation_data = extract_activations(
    data_loader=train_loader,
    model=model,
    sae=sae,
    device=device
)

print("\n--- 4. Generating Aux Loss Targets ---")
sparse_vector_sizes = [25, 5120]
for N_recon in sparse_vector_sizes:
    labels = activation_data["labels"]
    sparse_act_one = activation_data["sparse"]
    avg_digit_encoding, top_n_features = get_top_N_features(N_recon, sparse_act_one, labels)
    
    feature_indices_dict = {}
    for digit in range(0, 10):
        feature_indices_dict[digit] = top_n_features[digit]['indices']
    
    print("Features used:")
    print(len(feature_indices_dict[0]))
    
    recon_max_sparse_training, recon_max_sparse_ablated_training = get_sublabel_data(
                                                                    labels,
                                                                    feature_indices_dict,
                                                                    sparse_act_one,
                                                                    sae,
                                                                    device,
                                                                    FEATURE_DIM * 4
                                                                )
    
    file_path = f"./{N_recon}_top.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(recon_max_sparse_training, f)
    
    file_path = f"./{N_recon}_mask.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(recon_max_sparse_ablated_training, f)