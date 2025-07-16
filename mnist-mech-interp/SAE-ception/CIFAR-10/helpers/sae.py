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


def train_sae_on_layer(vit_model, target_layer, layer_name, sae_input_dim, train_loader, val_loader, device, sae_epochs, sae_l1_lambda, sae_save_path):
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
    vit_model.eval()
    best_val_loss = float('inf')

    for epoch in range(sae_epochs):
        # -- Training Phase --
        sae.train()
        total_loss = 0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Get activations from the ViT model without calculating gradients for it
            with torch.no_grad():
                vit_model(images)
            
            layer_activations = activations[layer_name]
            
            # SAE Forward and Backward Pass
            reconstructed_activations, hidden_activations = sae(layer_activations)
            reconstruction_loss = F.mse_loss(reconstructed_activations, layer_activations)
            l1_loss = torch.norm(hidden_activations, 1, dim=1).mean()
            loss = reconstruction_loss + sae_l1_lambda * l1_loss
            
            sae_optimizer.zero_grad()
            loss.backward()
            sae_optimizer.step()

            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, p=2, dim=1)

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # -- Validation Phase --
        sae.eval()
        total_val_loss = 0
        total_val_nonzero_activations = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                vit_model(images)
                layer_activations = activations[layer_name]
                
                reconstructed, hidden = sae(layer_activations)
                recon_loss = F.mse_loss(reconstructed, layer_activations)
                l1_loss = torch.norm(hidden, 1, dim=1).mean()
                loss = recon_loss + sae_l1_lambda * l1_loss
                total_val_loss += loss.item()

                total_val_nonzero_activations += torch.count_nonzero(hidden).item()

        avg_val_loss = total_val_loss / len(val_loader)
        num_val_samples = len(val_loader.dataset)
        avg_l0_norm = total_val_nonzero_activations / num_val_samples
        avg_l0_norm_percent = (avg_l0_norm / sae.hidden_dim) * 100
        print(f"  SAE Epoch [{epoch+1}/{sae_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Avg L0 Norm: {avg_l0_norm_percent:.2f}%")

        # -- Save Best Model --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(sae.state_dict(), sae_save_path)
            print(f"  ✨ New best SAE saved with validation loss: {avg_val_loss:.6f}")

    hook.remove()


def evaluate_sae_with_probe(vit_model, sae_model, target_layer, layer_name, train_loader, test_loader, device):
    """
    Evaluates a trained SAE by training a simple linear classifier (probe)
    on its sparse features and measuring the accuracy on the test set.
    """
    print(f"\n🔬 Evaluating SAE for layer '{layer_name}' with a linear probe...")

    # --- 1. Setup Probe ---
    probe_input_dim = sae_model.encoder.out_features
    num_classes = 10  # For CIFAR-10
    probe = nn.Linear(probe_input_dim, num_classes).to(device)
    probe_optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    probe_criterion = nn.CrossEntropyLoss()
    PROBE_EPOCHS = 10  # Number of epochs to train the probe

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

    # --- 3. Generate Sparse Features from the TRAINING set ---
    print("  Generating sparse features from the TRAINING set for probe training...")
    vit_model.eval()
    sae_model.eval()
    all_train_features = []
    all_train_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            vit_model(images)
            dense_activations = activations[layer_name]
            _, sparse_activations = sae_model(dense_activations)
            all_train_features.append(sparse_activations)
            all_train_labels.append(labels)
    
    train_features_tensor = torch.cat(all_train_features, dim=0)
    train_labels_tensor = torch.cat(all_train_labels, dim=0).to(device)
    probe_train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=64, shuffle=True)

    # --- 4. Train the Probe ---
    print(f"  Training the linear probe for {PROBE_EPOCHS} epochs...")
    probe.train()
    for epoch in range(PROBE_EPOCHS):
        for batch_features, batch_labels in probe_train_loader:
            probe_optimizer.zero_grad()
            outputs = probe(batch_features)
            loss = probe_criterion(outputs, batch_labels)
            loss.backward()
            probe_optimizer.step()

    # --- 5. Generate Sparse Features from the TEST set for evaluation ---
    print("  Generating sparse features from the TEST set for probe evaluation...")
    all_test_features = []
    all_test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            vit_model(images)
            dense_activations = activations[layer_name]
            _, sparse_activations = sae_model(dense_activations)
            all_test_features.append(sparse_activations)
            all_test_labels.append(labels)

    test_features_tensor = torch.cat(all_test_features, dim=0)
    test_labels_tensor = torch.cat(all_test_labels, dim=0).to(device)

    # --- 6. Evaluate the Probe on the Test Set ---
    probe.eval()
    with torch.no_grad():
        outputs = probe(test_features_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total = test_labels_tensor.size(0)
        correct = (predicted == test_labels_tensor).sum().item()

    accuracy = 100 * correct / total
    print(f"  🎯 Probe Accuracy on Test Set for '{layer_name}': {accuracy:.2f}%")

    # --- Calculate and report sparsity on the test set's sparse features ---
    with torch.no_grad():
        num_zero_activations = (test_features_tensor == 0).sum().item()
        total_activations = test_features_tensor.numel()
        sparsity = 100 * (num_zero_activations / total_activations)
        print(f"  📊 Sparsity on Test Set for '{layer_name}': {sparsity:.2f}% of hidden neurons were zero.")

    # --- 7. Cleanup ---
    hook.remove()
    return accuracy, sparsity