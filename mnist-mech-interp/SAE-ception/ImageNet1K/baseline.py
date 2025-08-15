import gc
import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from datasets import load_dataset
from tqdm import tqdm 

from helpers.helpers import set_seed
from helpers.sae import SparseAutoencoder, train_sae_on_layer, evaluate_sae_with_probe


# --- 0. For reproducibility & Configuration ---
SEED = 42
# baseline model accuracy is 87.89%
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
model.load_state_dict(torch.load(MODEL_LOAD_PATH))
model.to(device)
model.eval()

print("\n--- 2. Loading Data ---")
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

NUM_WORKERS = 4
test_split = load_dataset("ILSVRC/imagenet-1k", split='validation', cache_dir='./data')     # standard to use val as test for Image Net 1k
test_loader = DataLoader(test_split, batch_size=BATCH_SIZE, collate_fn=collate,
                         num_workers=NUM_WORKERS, pin_memory=True)

print("\n--- 3. Perfroming Eval ---")
correct = total = 0
test_bar = tqdm(test_loader, desc=f"Evaluating")
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {correct/total*100:.2f}%") 