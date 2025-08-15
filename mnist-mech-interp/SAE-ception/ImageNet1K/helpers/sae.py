import gc
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim_ratio=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * hidden_dim_ratio

        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        encoded = self.activation(self.encoder(x))
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded
    
    # def loss(self, x, reconstructed, encoded, l1_lambda=0.01):
    #     mse_loss = nn.MSELoss()(reconstructed, x)
    #     l1_loss = l1_lambda * torch.mean(torch.abs(encoded))
    #     return mse_loss + l1_loss


def train_sae_on_layer(model, target_layer, layer_name, sae_input_dim, train_loader, val_loader, device, sae_epochs, sae_l1_lambda, sae_save_path, val_set_size):
    """
    Trains an SAE on the activations of a specific layer of the ViT model.
    """
    print(f"\n🚀 Training SAE for layer: '{layer_name}'...")

    # --- SAE Setup ---
    SAE_LR = 1e-4
    
    sae = SparseAutoencoder(input_dim=sae_input_dim).to(device)
    sae_optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    
    # --- Hook Setup ---
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

    hook = target_layer.register_forward_hook(get_activation_hook(layer_name))

    # --- SAE Training Loop ---
    model.eval()
    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler()
    for epoch in range(sae_epochs):
        # -- Training Phase --
        sae.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{sae_epochs} [Train]", leave=False)
        for images, _ in train_bar:
            images = images.to(device)
            
            # Get activations from the model without calculating gradients for it
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    model(images)
                
                layer_activations = activations[layer_name]
                
                # SAE Forward and Backward Pass
                reconstructed_activations, hidden_activations = sae(layer_activations)
                reconstruction_loss = F.mse_loss(reconstructed_activations, layer_activations)
                l1_loss = torch.norm(hidden_activations, 1, dim=1).mean()
                loss = reconstruction_loss + sae_l1_lambda * l1_loss
            
            sae_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(sae_optimizer)
            scaler.update()

            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, p=2, dim=1)

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # -- Validation Phase --
        sae.eval()
        total_val_loss = 0
        total_val_nonzero_activations = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{sae_epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, _ in val_bar:
                images = images.to(device)

                model(images)
                layer_activations = activations[layer_name]

                # --- SAE Forward Pass ---
                reconstructed, hidden = sae(layer_activations)
                
                # --- Calculate Metrics for this Batch ---
                recon_loss = F.mse_loss(reconstructed, layer_activations)
                l1_loss = torch.norm(hidden, 1, dim=1).mean()
                loss = recon_loss + sae_l1_lambda * l1_loss
                
                # --- Accumulate Metrics ---
                total_val_loss += loss.item()
                total_val_nonzero_activations += torch.count_nonzero(hidden).item()

        # --- Calculate and Log Average Metrics for the Epoch ---
        avg_val_loss = total_val_loss / len(val_loader)
        avg_l0_norm = total_val_nonzero_activations / val_set_size
        avg_l0_norm_percent = (avg_l0_norm / sae.hidden_dim) * 100
        print(f"  SAE Epoch [{epoch+1}/{sae_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Avg L0 Norm: {avg_l0_norm_percent:.2f}%")

        # -- Save Best Model --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(sae.state_dict(), sae_save_path)
            print(f"  ✨ New best SAE saved with validation loss: {avg_val_loss:.6f}")

    hook.remove()


def evaluate_sae_with_probe(model, sae_model, target_layer, layer_name, train_loader, test_loader, device):
    """
    Evaluates a trained SAE by training a simple linear classifier (probe)
    on its sparse features and measuring the accuracy on the test set.
    """
    print(f"\n🔬 Evaluating SAE for layer '{layer_name}' with a linear probe...")

    # --- 1. Setup Probe ---
    probe_input_dim = sae_model.encoder.out_features
    num_classes = 1000
    probe = nn.Linear(probe_input_dim, num_classes).to(device)
    probe_optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    probe_criterion = nn.CrossEntropyLoss()
    # PROBE_EPOCHS = 5  # Number of epochs to train the probe
    PROBE_EPOCHS = 1  # Number of epochs to train the probe

    # --- 2. Hook Setup ---
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
    hook = target_layer.register_forward_hook(get_activation_hook(layer_name))

    # --- 3. Train the Probe (On-the-Fly) ---
    print(f"  Training the linear probe for {PROBE_EPOCHS} epochs...")
    model.eval()
    sae_model.eval()
    probe.train()
    for epoch in range(PROBE_EPOCHS):
        probe_train_bar = tqdm(train_loader, desc=f"Probe Epoch {epoch+1}/{PROBE_EPOCHS}", leave=False)
        for images, labels in probe_train_bar:
            images, labels = images.to(device), labels.to(device)

            # Generate features for this batch
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    model(images)
                    dense_activations = activations[layer_name]
                    _, sparse_activations = sae_model(dense_activations)

            # Train the probe on this batch's features
            probe_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = probe(sparse_activations)
                loss = probe_criterion(outputs, labels)

            loss.backward()
            probe_optimizer.step()
    
    # --- 4. Evaluate the Probe on the Test Set (On-the-Fly) ---
    print("  Evaluating the probe on the test set...")
    probe.eval()
    total_correct = 0
    total_samples = 0
    total_nonzero_activations = 0
    total_activations = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Probe Evaluation"):
            images, labels = images.to(device), labels.to(device)

            # Generate features for this batch
            model(images)
            dense_activations = activations[layer_name]
            _, sparse_activations = sae_model(dense_activations)
            
            # Get probe predictions
            outputs = probe(sparse_activations)
            _, predicted = torch.max(outputs.data, 1)

            # Update accuracy metrics
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # Update sparsity metrics
            total_nonzero_activations += torch.count_nonzero(sparse_activations).item()
            total_activations += sparse_activations.numel()

    accuracy = 100 * total_correct / total_samples
    sparsity_percent = 100 * (1.0 - (total_nonzero_activations / total_activations))
    
    print(f"  🎯 Probe Accuracy on Test Set for '{layer_name}': {accuracy:.2f}%")
    print(f"  📊 Sparsity on Test Set for '{layer_name}': {sparsity_percent:.2f}% of hidden neurons were zero.")

    # --- 5. Cleanup ---
    hook.remove()
    del probe, probe_optimizer, probe_criterion
    gc.collect()
    torch.cuda.empty_cache()

    return accuracy, sparsity_percent