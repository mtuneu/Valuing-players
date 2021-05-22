import extract_information as ei
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def get_values(match_id):
    values_df = pd.read_pickle('match_values/'+str(match_id)+'.pkl')

    return values_df


def player_value(player_id, values_df):

    player_columns = values_df[values_df['player_id'] == player_id]
    
    player_value = player_columns['total_value'].sum()

    return player_value


def all_players_value(players_df,values_df):

    player_ids = players_df['player_id'].to_list()
    
    player_values = []

    for id in player_ids:
        p_value = player_value(id, values_df)

        player_values.append({'player_id': id, 'value': p_value})

    player_values = pd.DataFrame(player_values)


    return player_values

def get_rating(players_values, players_df):
    player_minutes = players_df[['player_id', 'player_name', 'minutes_played']]

    players_values = pd.merge(players_values, player_minutes, on='player_id')

    players_values['rating'] = (90 / players_values['minutes_played']) * players_values['value'] 

    print(players_values)


if __name__ == '__main__':

    match_id = 7430

    players = ei.get_players(match_id)
    events_df = ei.get_simple_events(match_id)
    players_df = ei.player_games(events_df, players)

    values_df = get_values(match_id)

    players_values = all_players_value(players_df, values_df)


    get_rating(players_values, players_df)

    



