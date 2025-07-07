import torch
from torch import nn

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
    
    def loss(self, x, reconstructed, encoded, l1_lambda=0.01):
        mse_loss = nn.MSELoss()(reconstructed, x)
        l1_loss = l1_lambda * torch.mean(torch.abs(encoded))
        return mse_loss + l1_loss