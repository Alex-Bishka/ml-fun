import torch
import numpy as np
from tqdm import tqdm

def evaluate_and_gather_activations(model, sae_one, sae_two, data_loader, device):
    """
    Performs a full evaluation pass to compute metrics and gather activations.
    
    This function is designed to be memory-efficient by appending tensors to lists
    and concatenating them once at the end, avoiding slow per-batch .cpu() calls
    on pre-allocated tensors.
    """
    model.eval()
    sae_one.eval()
    sae_two.eval()
    
    # For metrics
    total_correct = 0
    total_samples = 0
    recon_errors_one = []
    recon_errors_two = []
    
    # For storing activations and labels
    # We'll store them on the CPU in lists and concatenate later
    all_encoded_one = []
    all_encoded_two = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating and Gathering Activations", leave=False):
            # In your original code, the validation and test loaders returned 2 items,
            # while the train loader returned 3. This function assumes 2 for simplicity
            # during evaluation. Modify if your test/val loaders also return acts_one.
            if len(batch) == 3:
                images, labels, _ = batch # Unpack but ignore the third item
            else:
                images, labels = batch
            
            images, labels = images.to(device), labels.to(device)

            # --- Main Model Forward Pass ---
            classification_out, hidden_one_act, hidden_two_act = model(images)
            
            # --- SAE Forward Pass ---
            reconstructed_one, encoded_one = sae_one(hidden_one_act)
            reconstructed_two, encoded_two = sae_two(hidden_two_act)
            
            # --- Store Activations and Labels (Efficiently) ---
            # Move to CPU now and append to lists. This is much faster than indexing.
            all_encoded_one.append(encoded_one.cpu())
            all_encoded_two.append(encoded_two.cpu())
            all_labels.append(labels.cpu())

            # --- Calculate Metrics ---
            # Accuracy
            _, predicted = torch.max(classification_out, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Reconstruction Error
            recon_errors_one.append(torch.mean((hidden_one_act - reconstructed_one) ** 2).item())
            recon_errors_two.append(torch.mean((hidden_two_act - reconstructed_two) ** 2).item())
            
    # --- Post-processing ---
    # Finalize metrics
    accuracy = 100 * total_correct / total_samples
    avg_recon_error_one = np.mean(recon_errors_one)
    avg_recon_error_two = np.mean(recon_errors_two)
    
    # Concatenate all stored tensors in one go
    Z_one = torch.cat(all_encoded_one, dim=0).numpy()
    Z_two = torch.cat(all_encoded_two, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    
    # Create a dictionary for clean return
    results = {
        "accuracy": accuracy,
        "recon_error_one": avg_recon_error_one,
        "recon_error_two": avg_recon_error_two,
        "Z_one": Z_one,
        "Z_two": Z_two,
        "y": y
    }
    
    return results