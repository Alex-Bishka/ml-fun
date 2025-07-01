import torch
import pickle
import random
import numpy as np
from tqdm import tqdm

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # These two lines are crucial for deterministic results on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# for reproducibility on training
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_sublabel_data(data_labels, data_images, feature_indices_dict, sparse_activations, sae_model, device, hidden_dims):
    recon_max_sparse = []
    recon_max_sparse_ablated = []
    for i, label in enumerate(data_labels):
        feature_indices = feature_indices_dict[label]
        image = torch.from_numpy(data_images[i]).float().unsqueeze(0).to(device)
        
        sparse_vector = sparse_activations[i].copy()
        encoded = np.zeros(hidden_dims)
        encoded_ablated = sparse_vector.copy()
        
        for feature_idx in feature_indices:
            encoded[feature_idx] = sparse_vector[feature_idx]
            encoded_ablated[feature_idx] = 0
        
        encoded_digit_torch = torch.from_numpy(encoded).float().to(device).unsqueeze(0)
        encoded_digit_torch_ablated = torch.from_numpy(encoded_ablated).float().to(device).unsqueeze(0)
        with torch.no_grad():
            recon = sae_model.decoder(encoded_digit_torch)
            recon = recon.view(1, -1)
    
            recon_ablated = sae_model.decoder(encoded_digit_torch_ablated)
            recon_ablated = recon.view(1, -1)
    
            recon_max_sparse.append(recon.cpu().detach().clone())
            recon_max_sparse_ablated.append(recon_ablated.cpu().detach().clone())

    return recon_max_sparse, recon_max_sparse_ablated
    
def get_top_N_features(N, sparse_act_one, labels):
    avg_digit_encoding = {}
    top_n_features = {}
    
    for digit in range(10):
        mask = labels == digit
        mean_digit_encoding = np.mean(sparse_act_one[mask], axis=0)
        avg_digit_encoding[digit] = mean_digit_encoding
    
        top_n_indices = np.argsort(mean_digit_encoding)[-N:][::-1]  # Sort descending
        top_n_values = mean_digit_encoding[top_n_indices]
    
        top_n_features[digit] = {
            'indices': top_n_indices.tolist(),
            'activations': top_n_values.tolist()
        }
    
    return avg_digit_encoding, top_n_features
    
def extract_activations(data_loader, model, sae_one, sae_two, device):
    """
    Extracts hidden, sparse, and reconstructed activations from a model and its SAEs.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        model (torch.nn.Module): The primary model to be evaluated.
        sae_one (torch.nn.Module): The first sparse autoencoder.
        sae_two (torch.nn.Module): The second sparse autoencoder.
        device (torch.device): The device to run the models on (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary containing numpy arrays for:
              'hidden_one', 'hidden_two', 'sparse_one', 'sparse_two',
              'recon_one', 'recon_two', and 'labels'.
    """
    # Ensure models are in evaluation mode
    model.eval()
    sae_one.eval()
    sae_two.eval()

    # Initialize lists to store activations and labels
    hidden_activations_one = []
    hidden_activations_two = []
    sparse_act_one = []
    sparse_act_two = []
    recon_act_one = []
    recon_act_two = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting Activations"):
            images = images.to(device)

            # Forward pass through the main model
            _, hidden_one_act, hidden_two_act = model(images)

            # Forward pass through the sparse autoencoders
            recon_one, encoding_one = sae_one(hidden_one_act)
            recon_two, encoding_two = sae_two(hidden_two_act)

            # Append results to lists
            hidden_activations_one.append(hidden_one_act.cpu().numpy())
            hidden_activations_two.append(hidden_two_act.cpu().numpy())
            sparse_act_one.append(encoding_one.cpu().numpy())
            sparse_act_two.append(encoding_two.cpu().numpy())
            recon_act_one.append(recon_one.cpu().numpy())
            recon_act_two.append(recon_two.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches and convert to numpy arrays
    results = {
        'hidden_one': np.concatenate(hidden_activations_one, axis=0),
        'hidden_two': np.concatenate(hidden_activations_two, axis=0),
        'sparse_one': np.concatenate(sparse_act_one, axis=0),
        'sparse_two': np.concatenate(sparse_act_two, axis=0),
        'recon_one': np.concatenate(recon_act_one, axis=0),
        'recon_two': np.concatenate(recon_act_two, axis=0),
        'labels': np.concatenate(all_labels, axis=0)
    }

    return results
    
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

def load_intermediate_labels(file_path):
    with open(file_path, "rb") as f:
        labels = pickle.load(f)

    return labels