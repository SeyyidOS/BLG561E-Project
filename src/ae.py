import torch.nn.functional as F
import torch.nn as nn
import torch


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


class MultiHeadAutoEncoder(nn.Module):
    def __init__(self, input_sizes, embedding_dim):
        super(MultiHeadAutoEncoder, self).__init__()
        
        self.league_head = nn.Sequential(
            nn.Linear(input_sizes["League Club History"], 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.national_head = nn.Sequential(
            nn.Linear(input_sizes["National Club History"], 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.league_history_head = nn.Sequential(
            nn.Linear(input_sizes["League History"], 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.attributes_head = nn.Sequential(
            nn.Linear(input_sizes["Attributes"], 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.categorical_head = nn.Sequential(
            nn.Linear(input_sizes["Categorical"], 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
        self.embedding_layer = nn.Linear(embedding_dim * 5, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, sum(input_sizes.values()))
        )

    def forward(self, league_club, national_club, league_history, attributes, categorical):
        league_embedding = self.league_head(league_club)
        national_embedding = self.national_head(national_club)
        league_history_embedding = self.league_history_head(league_history)
        attributes_embedding = self.attributes_head(attributes)
        categorical_embedding = self.categorical_head(categorical)
        
        combined = torch.cat(
            [league_embedding, national_embedding, league_history_embedding, attributes_embedding, categorical_embedding],
            dim=1
        )
        shared_embedding = self.embedding_layer(combined)
        
        reconstruction = self.decoder(shared_embedding)
        return shared_embedding, reconstruction
