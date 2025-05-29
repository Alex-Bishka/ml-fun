import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Graphing code to visualize weights
def plot_weights(model, epoch, experiment_type, run_id, save_plt=False):
    save_path_hidden_one = f'./weights/{experiment_type}/{run_id}/hidden_one/hidden_one_weights_{epoch + 1}.png'
    save_path_classification = f'./weights/{experiment_type}/{run_id}/classification/classification_weights_{epoch + 1}.png'
    save_path_hidden_two = f'./weights/{experiment_type}/{run_id}/hidden_two/hidden_two_weights_{epoch + 1}.png'

    for path in [save_path_hidden_one, save_path_hidden_two, save_path_classification]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Extract weights from each layer
    hidden_one_w = model.hidden_one.weight.detach().cpu().numpy()  # Shape: (16, 784)
    hidden_two_w = model.hidden_two.weight.detach().cpu().numpy()  # Shape: (16, 16)
    classification_w = model.classification_layer.weight.detach().cpu().numpy()  # Shape: (10, 16)

    # Figure 1: Hidden One Weights (4x4 grid of 28x28 images)
    fig1, axes1 = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        row, col = divmod(i, 4)
        neuron_w = np.abs(hidden_one_w[i].reshape(28, 28))  # Reshape to 28x28
        axes1[row, col].imshow(neuron_w, cmap='gray')
        axes1[row, col].set_title(f"H1 Neuron {i+1}")
        axes1[row, col].axis('off')
    plt.suptitle(f"Hidden One Weights - Epoch {epoch+1}")
    plt.tight_layout()

    if save_plt:
        plt.savefig(save_path_hidden_one)
        plt.close()
    else:
        plt.show()

    # Figure 2: Hidden Two Weights (4x4 grid of line plots)
    # TODO: change this to scatter plot (these should be disrete data points)
    fig2, axes2 = plt.subplots(4, 4, figsize=(12, 12))
    min_w = hidden_two_w.min()
    max_w = hidden_two_w.max()
    for i in range(16):
        row, col = divmod(i, 4)
        axes2[row, col].plot(hidden_two_w[i])
        axes2[row, col].set_title(f"H2 Neuron {i+1}")
        axes2[row, col].set_xlabel("From H1 Neuron")
        axes2[row, col].set_ylabel("Weight")
        axes2[row, col].set_ylim(min_w, max_w)
    plt.suptitle(f"Hidden Two Weights - Epoch {epoch+1}")
    plt.tight_layout()

    if save_plt:
        plt.savefig(save_path_hidden_two)
        plt.close()
    else:
        plt.show()

    # Figure 3: Classification Weights (2x5 grid of line plots)
    fig3, axes3 = plt.subplots(2, 5, figsize=(15, 6))
    min_w = classification_w.min()
    max_w = classification_w.max()
    for j in range(10):
        row, col = divmod(j, 5)
        axes3[row, col].plot(classification_w[j])
        axes3[row, col].set_title(f"Class {j}")
        axes3[row, col].set_xlabel("From H2 Neuron")
        axes3[row, col].set_ylabel("Weight")
        axes3[row, col].set_ylim(min_w, max_w)
    plt.suptitle(f"Classification Weights - Epoch {epoch+1}")
    plt.tight_layout()

    if save_plt:
        plt.savefig(save_path_classification)
        plt.close()
    else:
        plt.show()


def plot_losses(loss_one, loss_two, label_one, label_two):
    epochs = np.arange(1, len(loss_one) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_one, 'b-', label=f'{label_one} Loss')
    plt.plot(epochs, loss_two, 'r-', label=f'{label_two} Loss')
    plt.title(f'{label_one} and {label_two} Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_saliency_map(img_idx, feature_idx, classifier, sae, dataset, hidden_size, device, use_hidden_one=True, alpha=0.6, gradient_clipping=False):
    # idx of 3 is digit 0
    image, label = dataset[img_idx]  # Example image
    image = image.to(device).unsqueeze(0)  # Add batch dimension
    
    # Select a sparse feature (e.g., feature_idx=0)
    image.requires_grad = True
    
    # Compute gradient of the feature w.r.t. the image
    classifier.zero_grad()
    sae.zero_grad()
    _, hidden_one_act, hidden_two_act = classifier(image)

    if use_hidden_one:
        _, encoded = sae(hidden_one_act)
    else:
        _, encoded = sae(hidden_two_act)

    target_feature = encoded[0, feature_idx]
    target_feature.backward()

    # Clip gradients
    if gradient_clipping:
        max_grad_norm = 1.0  # Adjustable parameter
        torch.nn.utils.clip_grad_norm_([image], max_grad_norm)
    
    # Get the saliency map (absolute gradients)
    saliency = torch.abs(image.grad.data).cpu().numpy().squeeze()
    
    # Normalize for better visualization
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Prepare original image for overlay
    original_img = image.detach().cpu().numpy().squeeze()
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())  # Normalize to [0,1]
    
    # Visualize
    plt.figure(figsize=(15, 5))

    # og image
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Original Image (Label: {label})")
    plt.axis('off')

    # saliency map
    plt.subplot(1, 3, 2)
    plt.imshow(saliency, cmap='hot')
    plt.title(f"Saliency Map for Feature {feature_idx + 1}")
    plt.axis('off')

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(original_img, cmap='gray')
    plt.imshow(saliency, cmap='hot', alpha=alpha)  # Overlay with transparency
    plt.title(f"Overlay (Feature {feature_idx + 1})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_sparse_vecs_by_image(image, label, encoded_one, encoded_two, save_plt=False):
    fig = plt.figure(figsize=(10, 6))
    
    # Plot the input image
    plt.subplot(2, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Input Image (Label: {label})")
    plt.axis('off')
    
    # Plot sparse vector for sae_hidden_one
    plt.subplot(2, 2, 3)
    plt.bar(range(len(encoded_one)), encoded_one)
    plt.title("Sparse Vector (Hidden One)")
    plt.xlabel("Feature Index")
    plt.ylabel("Activation")
    plt.grid(True, alpha=0.3)
    
    # Plot sparse vector for sae_hidden_two
    plt.subplot(2, 2, 4)
    plt.bar(range(len(encoded_two)), encoded_two)
    plt.title("Sparse Vector (Hidden Two)")
    plt.xlabel("Feature Index")
    plt.ylabel("Activation")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if save_plt:
        path = f"./sparse_vectors/{EXPERIMENT_TYPE}/{HIDDEN_SIZE}-{L1_PENALTY}/{label}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_top_act_images_by_feature(feature_idx, feature_activations, images, labels, top_k=3):
    activations = feature_activations[feature_idx]
    top_indices = torch.topk(activations, top_k).indices
    print(f"Feature {feature_idx + 1} top {top_k} activations:")
    for idx in top_indices:
        image, label = images[idx], labels[idx]
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"Label: {label}  |  img idx: {idx}")
        plt.show()


def plot_activations(hidden_one_act, hidden_two_act, classification_out, label, save_plt=False, meta=None, experiment_type=None, run_id=None):
    if meta:
        label = f"{label}_{meta}"
        
    # Convert tensors to numpy arrays
    hidden_one_act_np = hidden_one_act.cpu().detach().numpy()
    hidden_two_act_np = hidden_two_act.cpu().detach().numpy()
    classification_act_np = classification_out.cpu().detach().numpy()
    
    # Squeeze any singleton dimensions (e.g., (1, 16) -> (16,))
    hidden_one_act_np = np.squeeze(hidden_one_act_np)
    hidden_two_act_np = np.squeeze(hidden_two_act_np)
    classification_act_np = np.squeeze(classification_act_np)
    
    def normalize(acts):
        # Handle scalar or single-value arrays
        if acts.size == 1:
            return np.array([0.5])  # Map to middle of colormap
            
        # Add epsilon to avoid division by zero
        return (acts - acts.min()) / (acts.max() - acts.min() + 1e-8)
        
    hidden_one_act_norm = normalize(hidden_one_act_np)
    hidden_two_act_norm = normalize(hidden_two_act_np)
    classification_act_norm = normalize(classification_act_np)
    
    fig, ax = plt.subplots(figsize=(6, 12))

    ax.set_facecolor('#ADD8E6')
    fig.patch.set_facecolor('#ADD8E6')
    
    for i in range(16):  # Hidden Layer 1
        ax.add_patch(plt.Circle((1, i), radius=0.5, color=plt.cm.gray(hidden_one_act_norm[i])))
        
    for i in range(16):  # Hidden Layer 2
        ax.add_patch(plt.Circle((2, i), radius=0.5, color=plt.cm.gray(hidden_two_act_norm[i])))
        
    for i in range(10):  # Output Layer
        ax.add_patch(plt.Circle((3, i), radius=0.5, color=plt.cm.gray(classification_act_norm[i])))
             
    ax.set_ylim(-1, 16)
    ax.set_xlim(0, 4)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Hidden 1', 'Hidden 2', 'Output'])
    ax.set_yticks(range(16))  # Explicit ticks for 0–15
    ax.set_yticklabels(range(16))
    ax.set_title(f"Neural Network Activation Visualization - {label}")

    if save_plt:
        path = f"./avg_activations/{experiment_type}/{run_id}/{label}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()