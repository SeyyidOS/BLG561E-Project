from tqdm import tqdm

import pandas as pd
import numpy as np
import re


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["height"] = data["height"].str.replace(" cm", "", regex=False).astype(float)
    data["weight"] = data["weight"].str.replace(" kg", "", regex=False).astype(float)

    data = pd.get_dummies(data, columns=["nationality", "position"], prefix=["nationality", "position"])

    year_columns = [col for col in data.columns if re.match(r'^\d{4}(-\d+)?$', col)]
    data = pd.get_dummies(data, columns=year_columns, prefix=year_columns)

    data = data.drop(columns=["id", "age", "height", "weight", "photo", "player_name", "", "-2"])
    
    return data

def generate_player_data(data: dict) -> tuple[pd.DataFrame, dict]:
    player_rows = {}
    team_id_to_name = {}

    for player_name, player_data in data.items():
        info = player_data.get("player_info", {})
        seasons_data = player_data.get("seasons_data", {})

        row = {
            "id": info.get("id", ""),
            "player_name": player_name,
            "age": info.get("age", ""),
            "nationality": info.get("nationality", ""),
            "height": info.get("height", ""),
            "weight": info.get("weight", ""),
            "position": info.get("position", ""),
            "photo": info.get("photo", ""),
        }

        season_tracker = {}

        for team, team_info in seasons_data.items():
            team_id = team_info["team_id"]

            if team_id not in team_id_to_name:
                team_id_to_name[team_id] = team

            for season in team_info["seasons"]:
                if season in season_tracker:
                    season_tracker[season] += 1
                    season_key = f"{season}-{season_tracker[season]}"
                else:
                    season_tracker[season] = 1
                    season_key = str(season)

                row[season_key] = team

        player_rows[player_name] = row

    df = pd.DataFrame(player_rows).T.fillna(np.nan)
    df.reset_index(drop=True, inplace=True)

    return df, team_id_to_name

def generate_player_data_v2 (data: dict) -> pd.DataFrame:
    players_data = []
    for player, details in tqdm(data.items(), desc="Processing Players"):
        player_info = details["player_info"]
        season_dict = {}
        
        for team, team_details in details["seasons_data"].items():
            for season in team_details["seasons"]:
                season_dict[f"{team}_{season}"] = 1  # Mark presence of the season
        
        player_row = {
            "Id": player_info["id"],
            "Player Name": player,
            "Firstname": player_info["firstname"],
            "Lastname": player_info["lastname"],
            "Age": player_info["age"],
            "Birth Date": player_info["birth"]["date"],
            "Birth Place": player_info["birth"]["place"],
            "Birth Country": player_info["birth"]["country"],
            "Nationality": player_info["nationality"],
            "Height": player_info["height"],
            "Weight": player_info["weight"],
            "Number": player_info["number"],
            "Position": player_info["position"]
        }
        player_row.update(season_dict)  # Add the season columns
        players_data.append(player_row)

    df_seasons = pd.DataFrame(players_data).fillna(0)
    df_seasons = pd.get_dummies(df_seasons, columns=["Position", "Nationality"], prefix=["Position", "Nationality"])
    for col in df_seasons.select_dtypes(include=['uint8', 'bool']).columns:
        df_seasons[col] = df_seasons[col].astype(float)

    df_seasons['Height'] = df_seasons['Height'].str.replace(' cm', '').astype(float)
    df_seasons['Weight'] = df_seasons['Weight'].str.replace(' kg', '').astype(float)

    return df_seasons

def generate_player_data_v3 (data: dict, team_metadata: pd.DataFrame) -> tuple[pd.DataFrame, list, list, list]:
    team_metadata = team_metadata.set_index("team_name")
    league_clubs = set(team_metadata[~team_metadata["nation"]].index)
    national_clubs = set(team_metadata[team_metadata["nation"]].index)
    league_map = team_metadata["league"].to_dict()

    league_club_features = set()
    national_club_features = set()
    league_features = set()

    print("Extracting unique features...")
    for details in tqdm(data.values(), desc="Feature Extraction"):
        for club, club_data in details["seasons_data"].items():
            for year in club_data["seasons"]:
                feature = f"{club}_{year}"
                if club in league_clubs:
                    league_club_features.add(feature)
                if club in national_clubs:
                    national_club_features.add(feature)
                if club in league_map:
                    league = league_map[club]
                    league_feature = f"{league}_{year}"
                    league_features.add(league_feature)

    league_club_features = sorted(league_club_features)
    national_club_features = sorted(national_club_features)
    league_features = sorted(league_features)

    def one_hot_encode(items, all_features_set):
        feature_indices = {feature: i for i, feature in enumerate(all_features_set)}
        vector = [0] * len(all_features_set)
        for item in items:
            if item in feature_indices:
                vector[feature_indices[item]] = 1
        return np.array(vector)

    player_names = []
    ages = []
    weights = []
    heights = []
    positions = []
    nationalities = []
    league_club_vectors = []
    national_club_vectors = []
    league_history_vectors = []

    print("Processing player data...")
    for player, details in tqdm(data.items(), desc="Player Processing"):
        player_names.append(player)
        player_info = details["player_info"]
        player_seasons = details["seasons_data"]

        ages.append(player_info["age"])
        weights.append(player_info["weight"])
        heights.append(player_info["height"])
        positions.append(player_info["position"])
        nationalities.append(player_info["nationality"])

        league_club_items = []
        national_club_items = []
        league_items = []

        for club, club_data in player_seasons.items():
            for year in club_data["seasons"]:
                feature = f"{club}_{year}"
                if feature in league_club_features:
                    league_club_items.append(feature)
                if feature in national_club_features:
                    national_club_items.append(feature)
                
                league_row = team_metadata.loc[club] if club in team_metadata.index else None
                if league_row is not None:
                    league = league_row["league"]
                    league_feature = f"{league}_{year}"
                    if league_feature in league_features:
                        league_items.append(league_feature)

        league_club_vectors.append(one_hot_encode(league_club_items, league_club_features))
        national_club_vectors.append(one_hot_encode(national_club_items, national_club_features))
        league_history_vectors.append(one_hot_encode(league_items, league_features))

    dataframe = pd.DataFrame({
        "Player": player_names,
        "Age": ages,
        "Weight": weights,
        "Height": heights,
        "Position": positions,
        "Nationality": nationalities,
        "League Club History": league_club_vectors,
        "National Club History": national_club_vectors,
        "League History": league_history_vectors,
    })

    dataframe['Height'] = dataframe['Height'].str.replace(' cm', '').astype(float)
    dataframe['Weight'] = dataframe['Weight'].str.replace(' kg', '').astype(float)

    return dataframe, league_club_features, national_club_features, league_features

def generate_base_player_data(data: dict) -> pd.DataFrame:
    processed_data = []

    for player, details in data.items():
        player_info = details["player_info"]
        yearly_teams = {}
        
        for team, team_details in details["seasons_data"].items():
            for season in team_details["seasons"]:
                if season not in yearly_teams:
                    yearly_teams[season] = []
                yearly_teams[season].append(team)
        
        player_entry = {
            "Id": player_info["id"],
            "Player Name": player_info["name"],
            "Firstname": player_info["firstname"],
            "Lastname": player_info["lastname"],
            "Age": player_info["age"],
            "Birth Date": player_info["birth"]["date"],
            "Birth Place": player_info["birth"]["place"],
            "Birth Country": player_info["birth"]["country"],
            "Nationality": player_info["nationality"],
            "Height": player_info["height"],
            "Weight": player_info["weight"],
            "Number": player_info["number"],
            "Position": player_info["position"]
        }
        
        for year, teams in yearly_teams.items():
            player_entry[str(year)] = teams
        
        processed_data.append(player_entry)

    df_yearly_teams = pd.DataFrame(processed_data)

    return df_yearly_teams

def compute_weighted_similarity(
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

    similarity_matrix = np.zeros((n, n))

    with tqdm(total=n * (n - 1) // 2, desc="Calculating Similarities") as pbar:
        for i in range(n):
            for j in range(i + 1, n):  
                # Vector-based similarities using dot product
                league_club_sim = np.dot(df["League Club History"].iloc[i], df["League Club History"].iloc[j])
                national_club_sim = np.dot(df["National Club History"].iloc[i], df["National Club History"].iloc[j])
                league_history_sim = np.dot(df["League History"].iloc[i], df["League History"].iloc[j])

                # Scalar-based similarities
                age_sim = 1 - abs(df["Age"].iloc[i] - df["Age"].iloc[j]) / max(df["Age"].max() - df["Age"].min(), 1)
                weight_sim = 1 - abs(df["Weight"].iloc[i] - df["Weight"].iloc[j]) / max(df["Weight"].max() - df["Weight"].min(), 1)
                height_sim = 1 - abs(df["Height"].iloc[i] - df["Height"].iloc[j]) / max(df["Height"].max() - df["Height"].min(), 1)
                position_sim = 1 if df["Position"].iloc[i] == df["Position"].iloc[j] else 0
                nationality_sim = 1 if df["Nationality"].iloc[i] == df["Nationality"].iloc[j] else 0

                # Weighted similarity calculation
                total_similarity = (
                    league_club_weight * league_club_sim +
                    national_club_weight * national_club_sim +
                    league_history_weight * league_history_sim +
                    age_weight * age_sim +
                    weight_weight * weight_sim +
                    height_weight * height_sim +
                    position_weight * position_sim +
                    nationality_weight * nationality_sim
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

                similarity_matrix[i, j] = total_similarity
                similarity_matrix[j, i] = total_similarity  # Symmetric matrix

                pbar.update(1)

    similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)
    return similarity_df

def get_max_similarity(similarity_df):
    idxmax_row = similarity_df.idxmax(axis=0).loc[similarity_df.max().idxmax()]  # Row index
    idxmax_col = similarity_df.max().idxmax()  # Column index
    print(f"Maximum value is at row: {idxmax_row}, column: {idxmax_col}")

def get_similarity_vector(similarity_df, idx, df_base, similar_data_num=10):
    similarity_values = similarity_df.loc[idx]
    
    sorted_similarity = similarity_values.sort_values(ascending=False)
    
    top_similar_indices = sorted_similarity.head(similar_data_num).index.tolist()
    
    top_similar_indices.insert(0, idx)
    
    similar_data = df_base.loc[top_similar_indices].copy()
    
    similar_data['similarity_score'] = sorted_similarity[top_similar_indices].values
    
    return similar_data
