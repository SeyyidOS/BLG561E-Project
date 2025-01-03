import pandas as pd


def generate_player_data(data: dict) -> pd.DataFrame:
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

                row[season_key] = team_id

        player_rows[player_name] = row

    df = pd.DataFrame(player_rows).T.fillna(0)
    df.reset_index(drop=True, inplace=True)

    return df, team_id_to_name
