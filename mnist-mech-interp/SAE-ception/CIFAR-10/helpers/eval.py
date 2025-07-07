import torch
import numpy as np

def evaluate_and_gather_activations(model, sae_first, sae_middle, sae_last, data_loader, device):
    captured_activations = {}
    def get_activation(name):
        def hook(model, input, output):
            captured_activations[name] = input[0].detach()
        return hook

    vit_h_14_num_layers = 32
    first_layer_index = 0
    middle_layer_index = vit_h_14_num_layers // 2 - 1
    last_layer_index = vit_h_14_num_layers - 1

    model.encoder.layers[first_layer_index].mlp.register_forward_hook(get_activation('first_mlp'))
    model.encoder.layers[middle_layer_index].mlp.register_forward_hook(get_activation('middle_mlp'))
    model.encoder.layers[last_layer_index].mlp.register_forward_hook(get_activation('last_mlp'))

    model.eval()
    sae_first.eval()
    sae_middle.eval()
    sae_last.eval()
    
    # For metrics
    total_correct = 0
    total_samples = 0
    recon_errors_first = []
    recon_errors_middle = []
    recon_errors_last = []
    
    # For storing activations and labels
    # We'll store them on the CPU in lists and concatenate later
    all_encoded_first = []
    all_encoded_middle = []
    all_encoded_last = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # --- Main Model Forward Pass ---
            classification_out = model(images)
            
            # --- SAE Forward Pass ---
            act_first = captured_activations['first_mlp']
            act_middle = captured_activations['middle_mlp']
            act_last = captured_activations['last_mlp']

            recon_first, encoded_first = sae_first(act_first)
            recon_middle, encoded_middle = sae_middle(act_middle)
            recon_last, encoded_last = sae_last(act_last)
            
            # --- Store Activations and Labels (Efficiently) ---
            # Move to CPU now and append to lists. This is much faster than indexing.
            all_encoded_first.append(encoded_first.cpu())
            all_encoded_middle.append(encoded_middle.cpu())
            all_encoded_last.append(encoded_last.cpu())
            all_labels.append(labels.cpu())

            # --- Calculate Metrics ---
            # Accuracy
            _, predicted = torch.max(classification_out, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Reconstruction Error
            recon_errors_first.append(torch.mean((act_first - recon_first) ** 2).item())
            recon_errors_middle.append(torch.mean((act_middle - recon_middle) ** 2).item())
            recon_errors_last.append(torch.mean((act_last - recon_last) ** 2).item())
            
    # --- Post-processing ---
    # Finalize metrics
    accuracy = 100 * total_correct / total_samples
    avg_recon_error_first = np.mean(recon_errors_first)
    avg_recon_error_middle = np.mean(recon_errors_middle)
    avg_recon_error_last = np.mean(recon_errors_last)
    
    # Concatenate all stored tensors in one go
    Z_first = torch.cat(all_encoded_first, dim=0).numpy()
    Z_middle = torch.cat(all_encoded_middle, dim=0).numpy()
    Z_last = torch.cat(all_encoded_last, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    
    # Create a dictionary for clean return
    results = {
        "accuracy": accuracy,
        "recon_error_first": avg_recon_error_first,
        "recon_error_middle": avg_recon_error_middle,
        "recon_error_last": avg_recon_error_last,
        "Z_first": Z_first,
        "Z_middle": Z_middle,
        "Z_last": Z_last,
        "y": y
    }
    
    return results
