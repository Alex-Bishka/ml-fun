import os
import numpy as np
import pickle
import torch
from torchvision import datasets, transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torch.utils.data import DataLoader
import timm.data

import argparse

parser = argparse.ArgumentParser(description='Run a hyperparameter experiment for the classifier.')
parser.add_argument('--base_lr', type=float, required=True, help='The base learning rate for the model.')
parser.add_argument('--base_decay', type=float, required=True, help='The base weight decay for the model.')
args = parser.parse_args()

from helpers.dataset import ActivationDataset
from helpers.helpers import set_seed, load_intermediate_labels


#  --- 0. For reproducibility ---
BATCH_SIZE = 8
ACCUMULATION_STEPS = 8
IMG_RES = 384
NUM_EPOCHS = 1
MODEL_LOAD_PATH = './classifiers/baseline/vit_h_99.37.pth'
RECON_ACT_BASE_PATH = "./features/classifier-99.37"
BASE_LR = args.base_lr
BASE_DECAY = args.base_decay

# Tuning for loss factor
# min_loss = 0.01
# max_loss = 0.51
# step = 0.05
# loss_factors = np.arange(min_loss, round(max_loss + step, 3), step)
loss_factors = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 1, 3])
print(len(loss_factors))
print(loss_factors)
print(f"Base LR: {BASE_LR}  |  Base decay: {BASE_DECAY}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 Starting fine-tuning...")
# for N, sparse_type in [(25, "top"), (25, "mask"), (256, "top"), (256, "mask")]:
for N, sparse_type in [(25, "top")]:
    print(f"\nStarting run for {N}-{sparse_type}!\n")

    recon_act_path = f"{RECON_ACT_BASE_PATH}/{N}_{sparse_type}.pkl"
    recon_act_raw = load_intermediate_labels(recon_act_path)

    ######################################################################################################
    # DATA INIT
    ######################################################################################################
    seed = 42
    generator = torch.Generator().manual_seed(seed)

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
    indices = torch.randperm(num_train, generator=generator).tolist()

    train_subset = torch.utils.data.Subset(full_train_dataset, indices[:split])
    val_subset = torch.utils.data.Subset(val_dataset_for_split, indices[split:])

    train_dataset_with_activations = ActivationDataset(train_subset, recon_act_raw)

    train_loader = DataLoader(train_dataset_with_activations, batch_size=BATCH_SIZE,
                    shuffle=True, generator=generator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                    shuffle=False, generator=generator, num_workers=4, pin_memory=True)

    mixup = timm.data.Mixup(
        mixup_alpha=1.0,  # Strength of MixUp
        cutmix_alpha=0.0,  # Disable CutMix (set to 1.0 if you want to include it)
        num_classes=10,
        prob=1.0,  # Apply MixUp to all batches
        label_smoothing=0.1,  # Optional: adds label smoothing
    )

    # Create the test DataLoader
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                    pin_memory=True, num_workers=4)

    loss_data_dict = {}
    for loss_factor in loss_factors:
        print("#" * 50)
        print(f"Loss factor: {loss_factor}\n\n")

        ######################################################################################################
        # MODELS INIT
        ######################################################################################################
        set_seed(42)
        # weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
        weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        model = vit_h_14(weights=None) 
        model.image_size = IMG_RES

        # Update model's positional embeddings
        model.encoder.pos_embedding = torch.nn.Parameter(torch.load('./embeds/pos_embed_linear_384_99.37.pth'))

        num_ftrs = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load(MODEL_LOAD_PATH))
        model.to(device)

        activations = {}
        def get_activation_hook(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    activation_tensor = output[0]
                else:
                    activation_tensor = output
                
                if activation_tensor.dim() == 3:
                    activations[name] = activation_tensor[:, 0, :].detach()
                else:
                    activations[name] = activation_tensor.detach()
            return hook
        target_layer = model.encoder
        layer_name = "last_layer"
        hook = target_layer.register_forward_hook(get_activation_hook(layer_name))

        # to only re-train the encoder params:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.encoder.parameters():
            param.requires_grad = True

        # close with 1e-4 and 0.05 | 5e-5 and 0.1 is eh | 1e-4 and .15 is eh
        # optimizer = torch.optim.SGD(model.encoder.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.1)
        
        optimizer = torch.optim.SGD(model.encoder.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=BASE_DECAY)
        # optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        def feature_loss_fn(a, b):
            # a, b: [batch_size, feature_dim]
            cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            return (1 - cos_sim).mean()
        # feature_loss_fn = torch.nn.CosineSimilarity()

        ######################################################################################################
        # TRAINING LOOP
        ######################################################################################################
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            # -- Training Phase --
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            scaler = torch.amp.GradScaler('cuda')

            for i, (images, labels, recon_act) in enumerate(train_loader):
                images, labels, recon_act = images.to(device), labels.to(device), recon_act.to(device)
                images, labels = mixup(images, labels)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    dense_activations = activations[layer_name]
                    feature_loss = (1 - feature_loss_fn(dense_activations, recon_act)).mean()
                    loss = criterion(outputs, labels) + (loss_factor * feature_loss)
                    loss = loss / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * ACCUMULATION_STEPS
                if (i + 1) % 100 == 0:
                    print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                    running_loss = 0.0

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
                model_path = f'./saved_models/{N}_{sparse_type}/lr-{BASE_LR}-decay-{BASE_DECAY}/best_model_lf_{round(loss_factor, 3)}.pth'
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

    data_dict_path = f"./hyperparams/lr-{BASE_LR}-decay-{BASE_DECAY}/loss_data_dict_{round(loss_factors[0], 3)}_to_{round(loss_factors[-1], 3)}_{N}_{sparse_type}.pkl"
    os.makedirs(os.path.dirname(data_dict_path), exist_ok=True)
    with open(data_dict_path, "wb") as f:
        pickle.dump(loss_data_dict, f)

    print(f"\nJust completed run for {N}-{sparse_type}!\n")