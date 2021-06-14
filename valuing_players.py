from value_players.value_to_players import get_best_ratings, get_values, all_players_value
from events_extraction.extract_events import get_competitions, get_games
from events_extraction import extract_information as ei
import pandas as pd


def get_all_players_rating(competitions='All', gender='male'):
    competitions_df = get_competitions()

    path = 'ratings/all'

    if competitions != 'All':
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

    return all_players_df

if __name__ == "__main__":
    competitions_df = get_competitions()['competition_name'].unique()
    valid = False

    while(not valid):
        print("Please select one of the following options/competitions")
        index = 1
        for comp in competitions_df:
            print(str(index) + ". " + comp)
            index+=1
        print(str(index) + ". All")
        
        selected = input()

        if selected in competitions_df or selected == "All":
            valid = True

        gender_selection = 'male'
        
        if selected == "All":
            print("Select gender:")
            print("1. Male")
            print("2. Female")

            gender_selection = input()

            gender_selection = gender_selection.lower()

            if gender_selection != ("male" or "female"):
                valid = False
        
        print("Select goalkeepers or field players:")
        print("1. Goalkeepers")
        print("2. Field players")

        p = input()

        p = p.lower()
        if(p == 'goalkeepers'):
            g = True
        else:
            g = False

        print("Select the number of players you want to see on screen, or All if you want to see all players. (Best values)")

        n = input()
        n = n.lower()
        if n != 'all':
            n = int(n)  
          
    all_players_df = get_all_players_rating(competitions=selected, gender=gender_selection)

    get_best_ratings(all_players_df, g, n_players=n)



