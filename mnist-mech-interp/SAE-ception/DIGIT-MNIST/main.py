import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


from models_and_data.nn import NeuralNetwork
from models_and_data.sae import SparseAutoencoder
from models_and_data.edgedataset import EdgeDataset
from models_and_data.model_helpers import evaluate_and_gather_activations, load_intermediate_labels, seed_worker, set_seed


# Parameters
HIDDEN_SIZE = 256
L1_PENALTY = 0.75

# target features from iteration 3
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F3/25_top_0.14_25_top_0.06_25_top_0.27"
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F3/25_top_0.14_25_top_0.06_25_top_0.18"

# target features from iteration 2
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F2/25_top_0.14_25_top_0.06"
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F2/256_top_0.07_25_top_0.08"
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F2/256_top_0.07_256_top_0.04"
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F2/25_mask_0.01_25_mask_0.06"
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F2/256_mask_0.29_256_mask_0.02"

# target features from iteration 1
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F1/256_top_0.07"
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F1/25_top_0.14"
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F1/25_mask_0.01"
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F1/256_mask_0.29"

# target features from classifier
# RECON_ACT_BASE_PATH = f"./SAE-Results/256-0.75/features/F0"

# Tuning for loss factor
min_loss = 0.01
max_loss = 0.30
step = 0.01
loss_factors = np.arange(min_loss, round(max_loss + step, 3), step)
print(len(loss_factors))
print(loss_factors)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"We will be using device: {device}")

# train data
train_images = load_intermediate_labels("./intermediate-labels/first_layer/train_images.pkl")
train_labels = load_intermediate_labels("./intermediate-labels/first_layer/train_labels.pkl")

# val data
val_images = load_intermediate_labels("./intermediate-labels/first_layer/val_images.pkl")
val_labels = load_intermediate_labels("./intermediate-labels/first_layer/val_labels.pkl")

# test data
test_images = load_intermediate_labels("./intermediate-labels/first_layer/test_images.pkl")
test_labels = load_intermediate_labels("./intermediate-labels/first_layer/test_labels.pkl")

RAND_SEED = 42
torch.manual_seed(RAND_SEED)

N_examples = len(train_images)  # e.g., 50000
feat_dim = 16  # your SAE target dim

rand_mat = torch.randn(N_examples, feat_dim).float()
rand_mat = rand_mat / rand_mat.norm(dim=1, keepdim=True)  # normalize rows
recon_max_sparse_act_one = [rand_mat[i].clone() for i in range(N_examples)]

# loop across all reconstruction targets
# for N in [25, 256]:
for N in [256]:
    # for sparse_type in ["top", "mask"]:
    for sparse_type in ["mask"]:
        print(f"\nStarting run for {N}-{sparse_type}!\n")

        # recon_act_path = f"{RECON_ACT_BASE_PATH}/{N}_{sparse_type}.pkl"
        # recon_max_sparse_act_one = load_intermediate_labels(recon_act_path)

        loss_data_dict = {}
        for loss_factor in loss_factors:
            print("#" * 50)
            print(f"Loss factor: {loss_factor}\n\n")
            ######################################################################################################
            # MODELS INIT
            ######################################################################################################
            
            # for reproducibility
            set_seed(42) 
            model = NeuralNetwork().to(device)

            # for reproducibility
            set_seed(42)
            sae_hidden_one = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)

            # for reproducibility
            set_seed(42)
            sae_hidden_two = SparseAutoencoder(input_size=16, hidden_size=HIDDEN_SIZE).to(device)

            # print(f"SAE weights: {sae_hidden_one.encoder.weight[0][:5].detach().cpu().numpy()}")
            # print(f"SAE weights: {sae_hidden_two.encoder.weight[0][:5].detach().cpu().numpy()}")

            classification_loss_fn = nn.CrossEntropyLoss()
            hidden_act_one_loss_fn = nn.CosineSimilarity()
            
            optimizer = torch.optim.Adam(model.parameters())
            optimizer_sae_hidden_one = torch.optim.Adam(sae_hidden_one.parameters())
            optimizer_sae_hidden_two = torch.optim.Adam(sae_hidden_two.parameters())
                
            
            ######################################################################################################
            # DATA INIT
            ######################################################################################################
            seed = 42
            generator = torch.Generator().manual_seed(seed)
            
            NUM_WORKERS = 4
            if device.type.lower() == "cpu":
                NUM_WORKERS = 0
            
            # training data
            train_dataset = EdgeDataset(train_images, train_labels, recon_max_sparse_act_one)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS,
                                    worker_init_fn=seed_worker, generator=generator, pin_memory=True)
            
            # validation data
            val_dataset = EdgeDataset(val_images, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)  # larger batch size for faster validation
            
            # test data
            test_dataset = EdgeDataset(test_images, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            
            ######################################################################################################
            # TRAINING LOOP
            ######################################################################################################
            num_epochs = 20    
            best_val_acc = 0.0
            best_val_loss = float('inf')
            for epoch in range(num_epochs):
                # --- Training Phase ---
                model.train()  # set the model to training mode - this is currently a no-op
                sae_hidden_two.train()
                sae_hidden_one.train()
                
                train_loss, sae_loss_one, sae_loss_two = 0.0, 0.0, 0.0
            
                train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
                for batch_idx, batch in enumerate(train_bar):
                    # deconstruct batch items
                    images, labels, acts_one = batch
                    images, labels, acts_one = images.to(device), labels.to(device), acts_one.to(device)
                    
                    # forward pass
                    optimizer.zero_grad()
                    classification_out, hidden_act_one, hidden_act_two = model(images)            
                    sub_loss = (1 - hidden_act_one_loss_fn(hidden_act_one, acts_one)).mean()
                    total_loss = classification_loss_fn(classification_out, labels) + loss_factor * (sub_loss)
                    total_loss.backward()
                    optimizer.step()
                    train_loss += total_loss.item()
                    train_bar.set_postfix(loss=total_loss.item())
            
                    # to prevent backprop on both graphs:
                    hidden_act_one_detached = hidden_act_one.detach()
                    hidden_act_two_detached = hidden_act_two.detach()
            
                    # SAE loss and backprop - hidden layer one
                    optimizer_sae_hidden_one.zero_grad()
                    reconstructed_one, encoded_one = sae_hidden_one(hidden_act_one_detached)
                    loss1 = sae_hidden_one.loss(hidden_act_one_detached,
                                                            reconstructed_one,
                                                            encoded_one,
                                                            l1_lambda=L1_PENALTY
                                                            )
                    loss1.backward()
                    optimizer_sae_hidden_one.step()
                    sae_loss_one += loss1.item()
                    
                    # SAE loss and backprop - hidden layer two
                    optimizer_sae_hidden_two.zero_grad()
                    reconstructed_two, encoded_two = sae_hidden_two(hidden_act_two_detached)
                    loss2 = sae_hidden_two.loss(hidden_act_two_detached,
                                                            reconstructed_two,
                                                            encoded_two,
                                                            l1_lambda=L1_PENALTY
                                                            )
                    loss2.backward()
                    optimizer_sae_hidden_two.step()
                    sae_loss_two += loss2.item()
            
                # --- Validation Phase ---
                model.eval()
                val_loss, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        # deconstruct
                        images, labels = batch
                        images, labels = images.to(device), labels.to(device)
            
                        classification_out, _, _ = model(images)
                        loss = classification_loss_fn(classification_out, labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(classification_out, 1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
            
                # epoch stats
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * correct / total
            
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    
                    model_path = f'./models_saved/{N}_{sparse_type}/best_model_lf_{round(loss_factor, 3)}.pth'
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    # Save all three model states in one file
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'sae_one_state_dict': sae_hidden_one.state_dict(),
                        'sae_two_state_dict': sae_hidden_two.state_dict(),
                    }, model_path)
                    best_model_path = model_path
                    print(f"  -> Saved best model checkpoint to {best_model_path}")
            
            
            ######################################################################################################
            # EVAL
            ######################################################################################################
            if best_model_path is None:
                print("No best model was saved. Skipping evaluation.")
                continue
                
            # Load the best models from the saved checkpoint
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            sae_hidden_one.load_state_dict(checkpoint['sae_one_state_dict'])
            sae_hidden_two.load_state_dict(checkpoint['sae_two_state_dict'])

            # This ensures the data order is reset and will match any future manual evaluation.
            print("\nRe-initializing DataLoaders for consistent evaluation...")
            seed = 42
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS,
                                    worker_init_fn=seed_worker, generator=generator, pin_memory=True)
            # test_loader doesn't need re-init since shuffle=False, but it's good practice for clarity.
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

            # --- Get Test Metrics and Activations in ONE PASS ---
            test_results = evaluate_and_gather_activations(model, sae_hidden_one, sae_hidden_two, test_loader, device)
            Z_test_one, Z_test_two, y_test = test_results["Z_one"], test_results["Z_two"], test_results["y"]
            print(f"\nFinal Test Accuracy: {test_results['accuracy']:.2f}%")
            print(f"Average Reconstruction Error (Hidden One): {test_results['recon_error_one']:.4f}")
            print(f"Average Reconstruction Error (Hidden Two): {test_results['recon_error_two']:.4f}")
            
            # Compute sparsity (average non-zero features per image)
            sparsity_one = np.mean(Z_test_one > 1e-5) * Z_test_one.shape[1]
            sparsity_two = np.mean(Z_test_two > 1e-5) * Z_test_two.shape[1]
            print(f"Average Non-Zero Features per Image (Hidden One): {sparsity_one:.2f}")
            print(f"Average Non-Zero Features per Image (Hidden Two): {sparsity_two:.2f}")

            # --- Get Training Activations for Probes in ONE PASS ---
            # We only need Z_one, Z_two, and y, so we can ignore the other returned values
            print("\nGathering activations from training set for probing...")
            train_results = evaluate_and_gather_activations(model, sae_hidden_one, sae_hidden_two, train_loader, device)
            Z_train_one, Z_train_two, y_train = train_results["Z_one"], train_results["Z_two"], train_results["y"]
            
            ######################################################################################################
            # SPARSE FEATURE PROBES
            ######################################################################################################
            print("\n--- Training Linear Probes ---")
            clf_one = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
            clf_one.fit(Z_train_one, y_train)
            acc_one = clf_one.score(Z_test_one, y_test)
            print(f"Linear Probe Accuracy (Hidden One): {acc_one:.2%}")
            
            clf_two = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
            clf_two.fit(Z_train_two, y_train)
            acc_two = clf_two.score(Z_test_two, y_test)
            print(f"Linear Probe Accuracy (Hidden Two): {acc_two:.2%}")

            # --- Log final results ---
            loss_data_dict[loss_factor] = {
                "Final_Accuracy": test_results['accuracy'],
                "S1_Probe_Acccuracy": acc_one,
                "S2_Probe_Acccuracy": acc_two,
            }

            # --- Cleanup ---
            del model, sae_hidden_one, sae_hidden_two, checkpoint
            del Z_train_one, Z_train_two, y_train, Z_test_one, Z_test_two, y_test
            del train_results, test_results, clf_one, clf_two
            torch.cuda.empty_cache()
            print()

        file_path = f"./loss_data_dict_{min_loss}_to_{round(loss_factors[-1], 3)}_{N}_{sparse_type}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(loss_data_dict, f)

        print(f"\nJust completed run for {N}-{sparse_type}!\n")
