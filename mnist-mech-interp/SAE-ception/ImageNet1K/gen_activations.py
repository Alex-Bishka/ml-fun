import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
from datasets import load_dataset


from helpers.sae import SparseAutoencoder
from helpers.helpers import set_seed, extract_activations, get_top_N_features, get_sublabel_data


# --- 0. For reproducibility & Configuration ---
SEED = 42
# MODEL_LOAD_PATH = './SAE-Results/results/baseline/baseline_classifier.pth'
# SAE_MODEL_PATH = "./sae_models/baseline/sae_last_layer_l1_0.0005_30.pth"
# output_dir = "act-baseline"

# MODEL_LOAD_PATH = './SAE-Results/results/F0/best_model_lf_0.01.pth'
# SAE_MODEL_PATH = "./sae_models/F0/sae_last_layer_l1_0.0002.pth"
# SAE_MODEL_PATH = "./sae_models/F0/sae_last_layer_l1_0.0005.pth"
# output_dir = "act-F0"

MODEL_LOAD_PATH = './SAE-Results/results/F1/best_model_lf_0.5.pth'
SAE_MODEL_PATH = "./sae_models/F1/sae_last_layer_l1_0.0001.pth"
output_dir = "act-F1"

BATCH_SIZE = 128

VAL_SET_SIZE = 25000
IMG_RES = 384
FEATURE_DIM = 1536
N_TOP_FEATURES = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

print("\n--- 1. Convnext Model ---")
model = timm.create_model(
    'convnextv2_large.fcmae_ft_in22k_in1k_384',
    pretrained=False,
    num_classes=1000  # ImageNet-1K
)
model.load_state_dict(torch.load(MODEL_LOAD_PATH))
model.to(device)
model.eval()

sae = SparseAutoencoder(input_dim=FEATURE_DIM).to(device)
sae.load_state_dict(torch.load(SAE_MODEL_PATH))


print("\n--- 2. Preparing Data for SAE Training and Validation ---")
full_train_split = load_dataset("ILSVRC/imagenet-1k", split='train', cache_dir='./data')
shuffled_train = full_train_split.shuffle(seed=SEED)

val_split = shuffled_train.select(range(VAL_SET_SIZE))
train_split = shuffled_train.select(range(VAL_SET_SIZE, len(shuffled_train)))
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

def collate_for_eval(batch):
    imgs = torch.stack([tfms(x['image']) for x in batch])
    labels = torch.tensor([x['label'] for x in batch])
    return imgs, labels

NUM_WORKERS = 4
train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, collate_fn=collate,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, collate_fn=collate_for_eval, 
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_split, batch_size=BATCH_SIZE, collate_fn=collate_for_eval,
                        num_workers=NUM_WORKERS, pin_memory=True)


print("\n--- 3. Extracting Activations (from train data -> model adaptation) ---")
activation_data = extract_activations(
    data_loader=train_loader,
    model=model,
    sae=sae,
    device=device,
    output_dir=output_dir
)

# print("\n--- 3. Extracting Activations (from test data -> metrics) ---")
# activation_data = extract_activations(
#     data_loader=test_loader,
#     model=model,
#     sae=sae,
#     device=device,
#     output_dir=output_dir
# )