import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.BatchNorm1d(latent_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 4),
            nn.BatchNorm1d(latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, input_dim)
        )
       
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed