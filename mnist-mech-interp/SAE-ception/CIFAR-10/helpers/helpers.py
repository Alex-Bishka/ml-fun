import torch
import pickle
import random
import numpy as np


def load_intermediate_labels(file_path):
    with open(file_path, "rb") as f:
        labels = pickle.load(f)

    return labels


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sublabel_data(data_labels, feature_indices_dict, sparse_activations, sae_model, device, hidden_dims):
    recon_max_sparse = []
    recon_max_sparse_ablated = []
    for i, label in enumerate(data_labels):
        feature_indices = feature_indices_dict[label]
        
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
            recon_ablated = recon_ablated.view(1, -1)
    
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
    

def extract_activations(data_loader, model, sae, device):
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
    sae.eval()

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

    # Initialize lists to store activations and labels
    hidden_activations_one = []
    sparse_act_one = []
    recon_act_one = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            # Forward pass through the main model
            model(images)

            dense_activations = activations[layer_name]

            # Forward pass through the sparse autoencoders
            recon_one, encoding_one = sae(dense_activations)

            # Append results to lists
            hidden_activations_one.append(dense_activations.cpu().numpy())
            sparse_act_one.append(encoding_one.cpu().numpy())
            recon_act_one.append(recon_one.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    hook.remove()

    # Concatenate all batches and convert to numpy arrays
    results = {
        'hidden_one': np.concatenate(hidden_activations_one, axis=0),
        'sparse_one': np.concatenate(sparse_act_one, axis=0),
        'recon_one': np.concatenate(recon_act_one, axis=0),
        'labels': np.concatenate(all_labels, axis=0)
    }

    return results