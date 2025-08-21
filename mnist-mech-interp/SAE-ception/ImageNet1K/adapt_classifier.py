import os
import numpy as np
import pickle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from datasets import load_dataset

from helpers.dataset import ChunkedActivationDataset
from helpers.helpers import set_seed


#  --- 0. For reproducibility ---
BATCH_SIZE = 64
NUM_EPOCHS = 1

# MODEL_LOAD_PATH = './SAE-Results/results/baseline/baseline_classifier.pth'
# ACTIVATIONS_BASE_PATH = "./SAE-Results/training-features/baseline/aux-activations-top-25"

# MODEL_LOAD_PATH = './SAE-Results/results/F0/best_model_lf_0.01.pth'
# ACTIVATIONS_BASE_PATH = "./SAE-Results/training-features/F0/aux-activations-top-25"
# RECON_KEY = "recon_top"

MODEL_LOAD_PATH = './SAE-Results/results/F1/best_model_lf_0.5.pth'
ACTIVATIONS_BASE_PATH = "./SAE-Results/training-features/F1/aux-activations-top-25"
RECON_KEY = "recon_top"

SEED = 42
VAL_SET_SIZE = 25000
IMG_RES = 384
FEATURE_DIM = 1536

BASE_LR = 1e-4
BASE_DECAY = 0.05

# Tuning for loss factor
# min_loss = 0.01
# max_loss = 0.51
# step = 0.05
# loss_factors = np.arange(min_loss, round(max_loss + step, 3), step)
loss_factors = np.array([0.01, 0.1, 0.2, 0.5])
print(len(loss_factors))
print(loss_factors)
print(f"Base LR: {BASE_LR}  |  Base decay: {BASE_DECAY}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)
print(f"\nStarting SAE-ception run!\n")

######################################################################################################
# DATA INIT
######################################################################################################
generator = torch.Generator().manual_seed(SEED)

full_train_split = load_dataset("ILSVRC/imagenet-1k", split='train', cache_dir='./data')
shuffled_train = full_train_split.shuffle(seed=SEED)

val_split = shuffled_train.select(range(VAL_SET_SIZE))
train_split = shuffled_train.select(range(VAL_SET_SIZE, len(shuffled_train)))
test_split = load_dataset("ILSVRC/imagenet-1k", split='validation', cache_dir='./data')     # standard to use val as test for Image Net 1k

train_dataset_with_activations = ChunkedActivationDataset(
    base_dataset=train_split,
    activations_dir=ACTIVATIONS_BASE_PATH,
    recon_key=RECON_KEY,
    shuffle_chunks=False,
    shuffle_in_chunk=False
)

set_seed(SEED)
temp_model = timm.create_model(
    'convnextv2_large.fcmae_ft_in22k_in1k_384',
    pretrained=False,
    num_classes=1000  # ImageNet-1K
)
temp_model.load_state_dict(torch.load(MODEL_LOAD_PATH))

tfms = transforms.Compose([
    transforms.Resize(IMG_RES),
    transforms.CenterCrop(IMG_RES),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=temp_model.default_cfg['mean'],
                        std=temp_model.default_cfg['std'])
])
del temp_model

def collate_for_train(batch):
    """
    Batch is list of tuples due to ChunkedDataset
    """
    images = torch.stack([tfms(item[0]) for item in batch])
    labels = torch.stack([item[1] for item in batch])
    recon_acts = torch.stack([item[2] for item in batch])
    return images, labels, recon_acts

def collate_for_eval(batch):
    imgs = torch.stack([tfms(x['image']) for x in batch])
    labels = torch.tensor([x['label'] for x in batch])
    return imgs, labels

NUM_WORKERS = 4
train_loader = DataLoader(train_dataset_with_activations, batch_size=BATCH_SIZE, collate_fn=collate_for_train,
                        num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, collate_fn=collate_for_eval, 
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_split, batch_size=BATCH_SIZE, collate_fn=collate_for_eval,
                        num_workers=NUM_WORKERS, pin_memory=True)

loss_data_dict = {}
for loss_factor in loss_factors:
    print("#" * 50)
    print(f"Loss factor: {loss_factor}\n\n")

    ######################################################################################################
    # MODELS INIT
    ######################################################################################################
    set_seed(SEED)

    print("\n--- Loading Convnext Model ---")
    model = timm.create_model(
        'convnextv2_large.fcmae_ft_in22k_in1k_384',
        pretrained=False,
        num_classes=1000  # ImageNet-1K
    )
    model.load_state_dict(torch.load(MODEL_LOAD_PATH))
    model.to(device)
    model.eval()

    activations = [] 
    def hook_fn(model, input, output):
        # No need to clear the list here, we will re-create it every time
        activation_tensor = output[0] if isinstance(output, tuple) else output
        if activation_tensor.dim() == 3:
            # Assuming shape is [batch, tokens, dim], we take the CLS token at index 0
            activations.append(activation_tensor[:, 0, :])
        else: # Assuming shape is [batch, dim]
            activations.append(activation_tensor)

    hook = model.head.flatten.register_forward_hook(hook_fn)

    # Freeze the entire model initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final classification layer (for class_loss)
    for param in model.head.fc.parameters():
        param.requires_grad = True

    # Unfreeze the normalization layer before the hook (for feature_loss)
    for param in model.head.norm.parameters():
        param.requires_grad = True

    trainable_params = list(model.head.fc.parameters()) + list(model.head.norm.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=BASE_LR, momentum=0.9, weight_decay=BASE_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    
    def feature_loss_fn(a, b):
        # a, b: [batch_size, feature_dim]
        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return (1 - cos_sim).mean()

    ######################################################################################################
    # TRAINING LOOP
    ######################################################################################################
    best_val_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        # -- Training Phase --
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for images, labels, recon_act in train_bar:
            activations = []
            optimizer.zero_grad()
            images, labels, recon_act = images.to(device), labels.to(device), recon_act.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                dense_activations = activations[0]

                class_loss = criterion(outputs, labels)
                feature_loss = feature_loss_fn(dense_activations, recon_act)

                loss = class_loss + (loss_factor * feature_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_bar.set_postfix(loss=loss.item())

        # -- Validation Phase --
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):  # Mixed precision for validation
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
            model_path = f'./saved_models/lr-{BASE_LR}-decay-{BASE_DECAY}/best_model_lf_{round(loss_factor, 3)}.pth'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
            print(f"✨ New best model saved with accuracy: {val_accuracy:.2f}% to {best_model_path}")

    hook.remove()
    print("✅ Fine-tuning finished.")


    # --- 5. Final Evaluation on Test Set ---
    print("\n🧪 Starting final evaluation on the test set...")

    # Load the best performing model
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    # Evaluate the model
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    final_accuracy = 100 * test_correct / test_total

    loss_data_dict[loss_factor] = {
        "Final_Accuracy": final_accuracy,
    }
    print(f"\n🎉 Final Accuracy of the best model on the test set: {final_accuracy:.2f}%")

data_dict_path = f'./saved_models/lr-{BASE_LR}-decay-{BASE_DECAY}/loss_data_dict_{round(loss_factors[0], 3)}_to_{round(loss_factors[-1], 3)}.pkl'
os.makedirs(os.path.dirname(data_dict_path), exist_ok=True)
with open(data_dict_path, "wb") as f:
    pickle.dump(loss_data_dict, f)

print(f"\nJust completed SAE-ception run!\n")