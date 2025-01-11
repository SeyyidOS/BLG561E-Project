def call_api(endpoint, params=None):
    if params is None:
        params = {}

    base_url = 'https://v3.football.api-sports.io/'
    url = f"{base_url}{endpoint}"
    headers = {
        'x-rapidapi-key': '52d7e411dc82fdcbbd5e857eb2a2a8b3',  # Replace with your API key
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        return response.json()  # Return parsed JSON response
    except requests.RequestException as e:
        print(f"API request error: {e}")
        return None


def players_data_dict(league, season, page=1, players_dict=None):
    if players_dict is None:
        players_dict = {}  # Initialize the dictionary

    params = {
        'league': league,
        'season': season,
        'page': page
    }

    players = call_api('players', params)

    if players:
        for player in players.get('response', []):
            player_id = player['player']['id']  # Extract the player ID
            player_name = player['player']['name']  # Extract the player name

            # Add the player only if the ID is not already in the dictionary
            if player_id not in players_dict:
                players_dict[player_id] = player_name  # Add player data with ID as the key

        # Check if more pages are available
        paging = players.get('paging', {})
        current_page = paging.get('current', 1)
        total_pages = paging.get('total', 1)

        if current_page < total_pages:
            next_page = current_page + 1
            if next_page % 2 == 1:
                time.sleep(1)  # Sleep for 1 second if the page is odd
            return players_data_dict(league, season, next_page, players_dict)

    return players_dict


# Function to fetch players for multiple leagues and seasons
def fetch_players_for_leagues_and_seasons(leagues, seasons):
    all_players_dict = {}  # To store all players data across all leagues and seasons
    for league in leagues:
        for season in seasons:
            print(f"Fetching players for League {league}, Season {season}...")
            players_dict = players_data_dict(league, season, players_dict=all_players_dict)
            all_players_dict.update(players_dict)  # Merge the data into the main dictionary
    return all_players_dict

