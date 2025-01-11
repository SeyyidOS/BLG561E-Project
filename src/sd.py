class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [
            os.path.join(image_folder, fname) for fname in os.listdir(image_folder)
            if fname.endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

def train_vae_step(vae, pixel_values):
    device = pixel_values.device

    vae.to(pixel_values.dtype)

    latents = vae.encode(pixel_values).latent_dist.sample() 

    reconstructed_images = vae.decode(latents).sample

    reconstructed_images = torch.clamp(reconstructed_images, min=-1.0, max=1.0)

    loss = torch.nn.functional.mse_loss(reconstructed_images, pixel_values)

    return loss



