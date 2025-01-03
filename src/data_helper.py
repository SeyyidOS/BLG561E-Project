import pandas as pd
import re


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["height"] = data["height"].str.replace(" cm", "", regex=False).astype(float)
    data["weight"] = data["weight"].str.replace(" kg", "", regex=False).astype(float)

    data = pd.get_dummies(data, columns=["nationality", "position"], prefix=["nationality", "position"])

    year_columns = [col for col in data.columns if re.match(r'^\d{4}(-\d+)?$', col)]
    data = pd.get_dummies(data, columns=year_columns, prefix=year_columns)

    data = data.drop(columns=["photo", "player_name"])
    
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

    df = pd.DataFrame(player_rows).T.fillna(0)
    df.reset_index(drop=True, inplace=True)

    return df, team_id_to_name
