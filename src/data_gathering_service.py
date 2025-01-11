def call_api(endpoint, params=None):
    if params is None:
        params = {}

    base_url = 'https://v3.football.api-sports.io/'
    url = f"{base_url}{endpoint}"
    headers = {
        'x-rapidapi-key': '52d7e411dc82fdcbbd5e857eb2a2a8b3',  
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() 
        return response.json()  
    except requests.RequestException as e:
        print(f"API request error: {e}")
        return None


def players_data_dict(league, season, page=1, players_dict=None):
    if players_dict is None:
        players_dict = {} 

    params = {
        'league': league,
        'season': season,
        'page': page
    }

    players = call_api('players', params)

    if players:
        for player in players.get('response', []):
            player_id = player['player']['id']  
            player_name = player['player']['name'] 

            if player_id not in players_dict:
                players_dict[player_id] = player_name  

        paging = players.get('paging', {})
        current_page = paging.get('current', 1)
        total_pages = paging.get('total', 1)

        if current_page < total_pages:
            next_page = current_page + 1
            if next_page % 2 == 1:
                time.sleep(1)  
            return players_data_dict(league, season, next_page, players_dict)

    return players_dict


def fetch_players_for_leagues_and_seasons(leagues, seasons):
    all_players_dict = {}  
    for league in leagues:
        for season in seasons:
            print(f"Fetching players for League {league}, Season {season}...")
            players_dict = players_data_dict(league, season, players_dict=all_players_dict)
            all_players_dict.update(players_dict) 
    return all_players_dict

