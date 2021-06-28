import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def get_values(match_id):
    """
    Loads a dataframe from a pickle with all the values from an specific match
    """
    values_df = pd.read_pickle('match_values/'+str(match_id)+'.pkl')

    return values_df


def player_value(player_id, values_df, is_gk):
    """
    Returns the value of an specific player
    """
    if is_gk == False:
        player_columns = values_df[values_df['player_id'] == player_id]
        
        player_value = player_columns['total_value'].sum()
    else:
        team_id = values_df.loc[0, 'team_id']
        player_columns = values_df[values_df['player_id'] == player_id]
        player_columns = player_columns.reset_index(drop=True)
        if player_columns.loc[0, 'team_id'] == team_id:
            player_value = player_columns.loc[0, 'team2_xG'] - player_columns.loc[0, 'team2_goals']
        else:
            player_value = player_columns.loc[0, 'team1_xG'] - player_columns.loc[0, 'team1_goals']


    return player_value


def all_players_value(players_df,values_df):
    """
    Returns a dataframe with the value for all players
    """
    gk = players_df[players_df['player_position'] == 'Goalkeeper']
    gk = gk['player_id'].to_list()
    player_ids = players_df['player_id'].to_list()
    player_values = []
    for id in player_ids:
        p_value = player_value(id, values_df, True if id in gk else False)

        player_values.append({'player_id': id, 'value': p_value})

    player_values = pd.DataFrame(player_values)


    return player_values

def get_rating(players_values, players_df):
    """
    Transform the players value to rating for 90 minutes.
    """
    player_minutes = players_df[['player_id', 'player_name', 'minutes_played']]

    players_values = pd.merge(players_values, player_minutes, on='player_id')

    players_values['rating'] = (90 / players_values['minutes_played']) * players_values['value'] 

    return players_values

def get_best_ratings(all_players_df, goalkeepers, n_players=10, minutes_played=900):
    """
    Prints a dataframe with the n best ratings, for the players or goalkeepers that played more than
    minutes_played
    """
    mask = all_players_df['minutes_played'] > minutes_played
    if goalkeepers:
        mask2 = all_players_df['player_position'] == 'Goalkeeper'
    else:
        mask2 = all_players_df['player_position'] != 'Goalkeeper'

    all_players_df = all_players_df[mask]
    all_players_df = all_players_df[mask2]
    

    all_players_df = all_players_df.sort_values('rating', ascending=False)

    if n_players == 'All':
        pd.set_option('display.max_rows', None)
        print(all_players_df)
    else:
        print(all_players_df.head(n_players))







    



