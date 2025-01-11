from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image

import streamlit as st
import torch
import json
import os


class ModelAPP:
    def __init__(self):
        self.image_folder = "./images/train"
        self.player_ids = None
        self.device = "cuda"


    def load_model(self, path):
        pass
    
    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")  
        return image

    def get_player_ids(self):
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        player_ids = [os.path.splitext(f)[0] for f in image_files]  
        return player_ids

    def blend(self, model, player1_id, player2_id, alpha):
        def postprocess_image(img):
            img = (img.clamp(-1, 1) + 1) / 2
            return img.permute(1, 2, 0).cpu().numpy() 

        image_transform = transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor(),         
            transforms.Normalize([0.5], [0.5])  
        ])

        image_path1 = os.path.join(self.image_folder, f"{player1_id}.png")
        image_path2 = os.path.join(self.image_folder, f"{player2_id}.png")
        image1 = self.load_image(image_path1)
        image2 = self.load_image(image_path2)


        processed_image1 = image_transform(image1).unsqueeze(0).to(self.device)
        processed_image2 = image_transform(image2).unsqueeze(0).to(self.device)



        with torch.no_grad():
            latent1 = model.encode(processed_image1).latent_dist.sample()
            latent2 = model.encode(processed_image2).latent_dist.sample()

            reconstructed_image1 = model.decode(latent1).sample[0]
            reconstructed_image2 = model.decode(latent2).sample[0]

            blended_latent = alpha * latent1 + (1 - alpha) * latent2

            blended_image = model.decode(blended_latent).sample[0]

        reconstructed_image1 = postprocess_image(reconstructed_image1)
        reconstructed_image2 = postprocess_image(reconstructed_image2)
        blended_image = postprocess_image(blended_image)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(reconstructed_image2, caption=f"Reconstructed {player2_id}", width=250)
        with col2:
            st.image(reconstructed_image1, caption=f"Reconstructed {player1_id}", width=250)
        with col3:
            st.image(blended_image, caption=f"Blended Image", width=250)

def main():
    with open('players_id_data.json') as f:
        id_name_map = json.load(f)

    name_id_map = {v: k for k, v in id_name_map.items()}
    
    options = [(v, k) for k, v in id_name_map.items()]
    

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.load_state_dict(torch.load("findetuned_sd_vae_v1.pth"))
    vae.eval()
    vae.to("cuda")

    app = ModelAPP()
    player_ids = app.get_player_ids()

    st.title("DALL-E Image Reconstruction and Blending by Player ID")
    st.header("Select Players for Image Blending")
    

    player2_id, selected_key = st.selectbox("Select a Player:", options, key="Player2")
    player1_id, selected_key = st.selectbox("Select a Player:", options, key="Player1")


    alpha = st.slider("Blend Ratio (Alpha)", 0.0, 1.0, 0.5)

    app.blend(vae, name_id_map[player1_id], name_id_map[player2_id], alpha)

if __name__ == "__main__":
    main()