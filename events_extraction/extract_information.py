import pandas as pd
import warnings
import json

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def get_simple_events(match_id):
    """
    Returns all the original StatsBomb events from an specific match
    """
    with open('./data/events/' + str(match_id) + '.json', encoding='utf-8') as json_file:
        data = json.load(json_file)
    events_df = pd.DataFrame(data)

    return events_df

def starting_lineup(match_id):
    """
    Returns the starting lineup of both teams from an specific match
    """
    with open('./data/lineups/'+str(match_id)+'.json', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    lineup_df = pd.DataFrame(data)

    return lineup_df

def get_players(match_id):
    """
    Returns all players starting from both team from an specific match
    """
    player_list = []
    for index, row in starting_lineup(match_id).iterrows():
        for p in row['lineup']:
            player_list.append({'player_id':p['player_id'], 'player_name': p['player_name']})
    
    player_df = pd.DataFrame(player_list)

    return player_df


def player_games(events_df, first_players_df):
    """
    Returns a dataframe with all the player games
    """
    max_minutes = max(events_df[events_df['type'] == {'id':34 , 'name': 'Half End'}].minute)
    
    players = []

    for index, row in events_df[events_df['type'] == {'id': 35, 'name': 'Starting XI'}].iterrows():
        for key, value in row['tactics'].items():
            if type(value) == list:
                for p in value:
                    player = {'player_id': p['player']['id'], 'player_name': p['player']['name'], 'minutes_played': max_minutes, 'is_starter': 1 if p['position'] else 0, 'player_position': p['position'].get('name')}
                    players.append(player)

    for index, row in events_df[events_df['type'] == {'id': 19, 'name': 'Substitution'}].iterrows():
        replacement = {'player_id': row['substitution']['replacement']['id'], 'player_name': row['substitution']['replacement']['name'], 'minutes_played': max_minutes - row['minute'], 'player_position': row['position'].get('name')}
        players.append(replacement)

    
    players_df = pd.DataFrame(players)

    players_df = players_df.fillna(0.0)

    players_df = pd.merge(players_df, first_players_df, on='player_id')

    players_df = players_df.drop(columns=['player_name_y'])

    players_df = players_df.rename(columns={'player_name_x' : 'player_name'})

    return players_df
