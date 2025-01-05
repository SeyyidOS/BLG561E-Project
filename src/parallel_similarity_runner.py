import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json 
from data_helper import generate_player_data_v3, generate_base_player_data

def calculate_similarity_chunk(chunk):
    i, df, weights, ranges, dot_products = chunk
    n = len(df)
    league_club_weight, national_club_weight, league_history_weight, age_weight, weight_weight, height_weight, position_weight, nationality_weight = weights
    age_range, weight_range, height_range = ranges
    league_club_sim_matrix, national_club_sim_matrix, league_history_sim_matrix = dot_products

    result = []
    for j in range(i + 1, n):
        # Calculate the weighted similarity
        total_similarity = (
            league_club_weight * league_club_sim_matrix[i, j] +
            national_club_weight * national_club_sim_matrix[i, j] +
            league_history_weight * league_history_sim_matrix[i, j] +
            age_weight * (1 - abs(df["Age"].iloc[i] - df["Age"].iloc[j]) / age_range) +
            weight_weight * (1 - abs(df["Weight"].iloc[i] - df["Weight"].iloc[j]) / weight_range) +
            height_weight * (1 - abs(df["Height"].iloc[i] - df["Height"].iloc[j]) / height_range) +
            position_weight * (1 if df["Position"].iloc[i] == df["Position"].iloc[j] else 0) +
            nationality_weight * (1 if df["Nationality"].iloc[i] == df["Nationality"].iloc[j] else 0)
        ) / (
            league_club_weight +
            national_club_weight +
            league_history_weight +
            age_weight +
            weight_weight +
            height_weight +
            position_weight +
            nationality_weight
        )
        result.append((i, j, total_similarity))
    return result

def compute_weighted_similarity_parallel(
    df,
    league_club_weight=1.0,
    national_club_weight=1.0,
    league_history_weight=1.0,
    age_weight=0.5,
    weight_weight=0.5,
    height_weight=0.5,
    position_weight=1.0,
    nationality_weight=1.0,
):
    n = len(df)

    # Precompute ranges and dot products
    age_range = max(df["Age"].max() - df["Age"].min(), 1)
    weight_range = max(df["Weight"].max() - df["Weight"].min(), 1)
    height_range = max(df["Height"].max() - df["Height"].min(), 1)

    league_club_history = np.array(df["League Club History"].tolist())
    national_club_history = np.array(df["National Club History"].tolist())
    league_history = np.array(df["League History"].tolist())

    league_club_sim_matrix = np.dot(league_club_history, league_club_history.T)
    national_club_sim_matrix = np.dot(national_club_history, national_club_history.T)
    league_history_sim_matrix = np.dot(league_history, league_history.T)

    weights = (league_club_weight, national_club_weight, league_history_weight, age_weight, weight_weight, height_weight, position_weight, nationality_weight)
    ranges = (age_range, weight_range, height_range)
    dot_products = (league_club_sim_matrix, national_club_sim_matrix, league_history_sim_matrix)

    # Prepare chunks for parallel processing
    chunks = [(i, df, weights, ranges, dot_products) for i in range(n)]

    # Use multiprocessing to calculate similarities
    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(calculate_similarity_chunk, chunks),
                total=n,
                desc="Calculating Similarities",
                mininterval=0.01  # Update tqdm every 0.01 seconds
            )
        )

    # Combine results into a similarity matrix
    similarity_matrix = np.zeros((n, n))
    for result in results:
        for i, j, sim in result:
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric matrix

    similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)
    return similarity_df



def main():
    with open('../data/player_details_v2.json') as f:
        data = json.load(f)

    team_metadata = pd.read_csv("../data/team_id_league_types.csv")

    df_base = generate_base_player_data(data)
    df, league_club_features, national_club_features, league_features = generate_player_data_v3(data, team_metadata)
    df = df.dropna()
    print("Started")
    similarity_df = compute_weighted_similarity_parallel(
        df[:4000],
        league_club_weight=5.0,
        national_club_weight=1.5,
        league_history_weight=1.0,
        age_weight=0.1,
        weight_weight=0.1,
        height_weight=0.1,
        position_weight=1.0,
        nationality_weight=1.5
    )

    # similarity_df.to_parquet('output.parquet', engine='pyarrow', index=False)
    similarity_df.to_csv('../data/similarity_df.csv', index=False)


if __name__ == "__main__":
    main()