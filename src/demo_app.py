from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
import streamlit as st
import pandas as pd
import random
import torch
import json
import os

@st.cache_resource
def load_model():
    model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    model.load_state_dict(torch.load("findetuned_sd_vae_v1.pth"))
    model.eval()
    model.to("cuda")
    return model

@st.cache_resource
def get_player_ids(csv_file):
    similarity_df = pd.read_csv(csv_file)
    similarity_df = similarity_df.set_index("Player")
    return [int(elem) for elem in similarity_df.index.to_list()]

@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

def refresh_csv(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_player_data(json_path):
    with open(json_path) as f:
        id_name_map = json.load(f)
    return id_name_map

class ModelAPP:
    def __init__(self, image_folder, similarity_csv):
        self.image_folder = image_folder
        self.device = "cuda"
        self.similarity_df = pd.read_csv(similarity_csv)
        self.similarity_df = self.similarity_df.set_index("Player")

        self.id_to_idx = pd.read_csv("id_to_index.csv")

    def load_image(self, image_path):
        return Image.open(image_path).convert("RGB")

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

        col2, col3 = st.columns(2)
        with col2:
            st.image(reconstructed_image1, caption=f"Selected Player {player1_id}", width=250)
        with col3:
            st.image(blended_image, caption=f"Hint", width=250)

    def generate_target_id(self):
        self.target_player_id = random.randint(0, 2)
        return self.target_player_id

    def get_similarity_rank(self, player_id):
        sorted_df = self.similarity_df.loc[self.target_player_id].to_frame().sort_values(by=self.target_player_id, ascending=False)
        indicies = sorted_df.index.to_list()
        return indicies.index(player_id) + 1

def main():
    games = [16]

    image_folder = "./images/train"
    csv_file = "similarities_final.csv"
    json_file = "players_id_data.json"

    player_ids = get_player_ids(csv_file)

    vae = load_model()

    app = ModelAPP(image_folder, csv_file)

    if "csv_data" not in st.session_state:
        st.session_state.csv_data = load_csv(csv_file)

    if st.button("Refresh CSV"):
        st.session_state.csv_data = refresh_csv(csv_file)
        st.success("CSV data reloaded!")

    id_name_map = load_player_data(json_file)
    name_id_map = {v: k for k, v in id_name_map.items() if int(k) in player_ids}
    options = [(v, k) for k, v in name_id_map.items()]
    

    st.title("Football Player Guessing Game")
    st.header("Find today's player. Start guessing!")

    game_id = st.selectbox("Select Game:", [i for i in range(1)], key="Game")
    app.target_player_id = games[int(game_id)]

    player1_id, selected_key = st.selectbox("Select a Player:", options, key="Player1")
    rank = app.get_similarity_rank(player1_id)
    st.text(f"Similarity Rank: {app.get_similarity_rank(player1_id)} ({player1_id})")


    alpha = (rank / (len(player_ids) * 2)) + 0.5 
    app.blend(vae, player1_id, games[game_id], alpha)

if __name__ == "__main__":
    main()
