from extract_events import get_competitions, get_games
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

    return players_values

def get_all_players_rating(competitions='all', gender='male'):
    competitions_df = get_competitions()

    path = 'ratings/all'

    if competitions != 'all':
        if type(competitions) == str:
            competitions_df = competitions_df[competitions_df['competition_name'] == competitions]
            path = competitions

    
    competitions_df = competitions_df[competitions_df['competition_gender'] == gender]

    path = path + '_' + gender

    all_players_df = pd.DataFrame(columns=['player_id', 'value','player_name','minutes_played'])
    
    for index, row in competitions_df.iterrows():
        competition_id = row['competition_id']
        season_id = row['season_id']

        games_df = get_games(competition_id, season_id)

        for _index, _row in games_df.iterrows():
            match_id = _row['match_id']

            players = ei.get_players(match_id)
            events_df = ei.get_simple_events(match_id)
            players_df = ei.player_games(events_df, players)

            values_df = get_values(match_id)

            players_values = all_players_value(players_df, values_df)

            players_df = players_df[['player_id', 'player_name','minutes_played', 'player_position']]
            players_values = players_values.merge(players_df, on='player_id')

            for i, r in players_values.iterrows():
                if r['player_id'] in all_players_df.values:
                    index = all_players_df[all_players_df['player_id'] == r['player_id']].index.values
                    old_value = all_players_df.loc[index, 'value']
                    new_value = old_value + r['value']
                    all_players_df.loc[index, 'value'] = new_value
                    old_minutes = all_players_df.loc[index, 'minutes_played']
                    new_minutes = old_minutes + r['minutes_played']
                    all_players_df.loc[index, 'minutes_played'] = new_minutes
                else:
                    all_players_df = all_players_df.append(r)
                all_players_df = all_players_df.reset_index(drop=True)
    all_players_df['rating'] = (90 / all_players_df['minutes_played']) * all_players_df['value']
        
    all_players_df.to_pickle(str(path)+'_players_rating.pkl')
    

    return all_players_df


def get_best_ratings(all_players_df, n_players=10, minutes_played=900):

    mask = all_players_df['minutes_played'] > minutes_played
    mask2 = all_players_df['player_position'] != 'Goalkeeper'

    all_players_df = all_players_df[mask]
    all_players_df = all_players_df[mask2].reset_index(drop=True)

    all_players_df = all_players_df.sort_values('rating', ascending=False)

    #print(all_players_df.head(n_players))

    pd.set_option('display.max_rows', None)

    print(all_players_df)
    

if __name__ == '__main__':

    """match_id = 15998

    players = ei.get_players(match_id)
    events_df = ei.get_simple_events(match_id)
    players_df = ei.player_games(events_df, players)
    print(players_df)
    values_df = get_values(match_id)

    players_values = all_players_value(players_df, values_df)

    print(get_rating(players_values, players_df))"""

    #all_players_df = get_all_players_rating()
    
    all_players_df = pd.read_pickle('ratings/all_male_players_rating.pkl')
    get_best_ratings(all_players_df)




    

    



