
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch
import os

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=1)
        )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Pixel değerleri [0, 1] aralığında olacak
        )
        
    def forward(self, x):
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, num_embeddings=512):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)
        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
    
    def forward(self, x):
        z_e = self.encoder(x)  # Latent encoding
        z_e_flattened = z_e.view(z_e.size(0), -1, z_e.size(1))
        
        distances = torch.cdist(z_e_flattened, self.codebook.weight, p=2)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices).view_as(z_e)
        
        x_reconstructed = self.decoder(z_q)
        return x_reconstructed, indices

class FootballPlayerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Görsel dosyalarını filtrele (jpg, png vb.)
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Görseli yükle ve RGB formatına çevir
        if self.transform:
            image = self.transform(image)
        return image, 0  # Sınıf etiketi tek bir değer olabilir (örneğin 0)

def train(model, dataloader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            reconstructed_images, _ = model(images)
            
            # MSE kaybı (reconstruction loss)
            loss = nn.functional.mse_loss(reconstructed_images, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
        visualize_reconstruction(model, images)

def visualize_reconstruction(model, images):
    model.eval()
    with torch.no_grad():
        reconstructed_images, _ = model(images)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(images[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed_images[0].permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Reconstructed")
    plt.show()
