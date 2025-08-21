import gc
import os
import torch
import pickle
import random
import numpy as np
import plotly.express as px


def SNE_plot_2d(activations_2d, labels, cluster_labels, width=1200, height=1200):
    fig = px.scatter(
        x=activations_2d[:, 0],
        y=activations_2d[:, 1],
        color=labels.astype(str),  # Color by true digit labels
        symbol=cluster_labels.astype(str),  # Different symbols for clusters
        labels={'color': 'Digit', 'symbol': 'Cluster'},
        title='t-SNE of Hidden Layer 1 Activations with K-Means Clustering',
        hover_data={'Digit': labels, 'Cluster': cluster_labels}
    )
    
    fig.update_layout(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        showlegend=True,
        coloraxis_colorbar_title='Digit',
        width=width,
        height=height
    )
    
    fig.write_json("temp.json")


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
    

def extract_activations(data_loader, model, sae, device, output_dir, batches_per_chunk=64):
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

    chunk_hidden_tensors = []
    chunk_sparse_tensors = []
    chunk_recon_tensors = []
    chunk_label_tensors = []

    captured_activation = []
    def hook_fn(model, input, output):
        activation_tensor = output[0] if isinstance(output, tuple) else output
        if activation_tensor.dim() == 3:
            captured_activation.append(activation_tensor[:, 0, :].detach())
        else:
            captured_activation.append(activation_tensor.detach())

    hook = model.head.flatten.register_forward_hook(hook_fn)

    os.makedirs(output_dir, exist_ok=True)
    chunk_counter = 0
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            
            captured_activation.clear()
            model(images)
            dense_activations = captured_activation[0]
            recon_act, sparse_act = sae(dense_activations)

            chunk_hidden_tensors.append(dense_activations.cpu())
            chunk_sparse_tensors.append(sparse_act.cpu())
            chunk_recon_tensors.append(recon_act.cpu())
            chunk_label_tensors.append(labels.cpu())

            # If the chunk is full (or it's the last batch)
            if (i + 1) % batches_per_chunk == 0 or (i + 1) == len(data_loader):
                print(f"Processing and saving chunk {chunk_counter}...")

                # Concatenate the chunk on the GPU
                hidden_chunk = torch.cat(chunk_hidden_tensors, dim=0).numpy()
                sparse_chunk = torch.cat(chunk_sparse_tensors, dim=0).numpy()
                recon_chunk = torch.cat(chunk_recon_tensors, dim=0).numpy()
                label_chunk = torch.cat(chunk_label_tensors, dim=0).numpy()

                # Save the numpy arrays to a compressed file
                chunk_filename = os.path.join(output_dir, f'chunk_{chunk_counter}.npz')
                np.savez_compressed(
                    chunk_filename,
                    hidden=hidden_chunk,
                    sparse=sparse_chunk,
                    recon=recon_chunk,
                    labels=label_chunk
                )
                
                chunk_counter += 1
                
                # Clear the GPU lists to free VRAM for the next chunk
                chunk_hidden_tensors.clear()
                chunk_sparse_tensors.clear()
                chunk_recon_tensors.clear()
                chunk_label_tensors.clear()
                
                # Optional: Force Python's garbage collector and empty PyTorch's cache
                gc.collect()

    hook.remove()
    print(f"Finished. All activation chunks saved in '{output_dir}'.")