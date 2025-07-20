import torch
from torch.utils.data import Dataset

# 1. Define the Custom Dataset
class ActivationDataset(Dataset):
    def __init__(self, subset, activations):
        """
        Args:
            subset (torch.utils.data.Subset): The subset containing images and labels.
            activations (numpy array or tensor): The corresponding reconstructed activations.
        """
        self.subset = subset
        
        if isinstance(activations, list):
            self.activations = torch.cat(activations, dim=0)
        elif not isinstance(activations, torch.Tensor):
            self.activations = torch.tensor(activations, dtype=torch.float32)
        else:
            self.activations = activations

        if len(self.subset) != len(self.activations):
            raise ValueError("Subset and activations must have the same length.")

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get image and label from the original subset
        image, label = self.subset[idx]
        # Get the corresponding activation
        activation = self.activations[idx]
        return image, label, activation