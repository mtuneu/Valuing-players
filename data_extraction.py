import extract_events
import events_to_actions as to_actions
import shots
from expected_goals import logistic_expected_goals
import pandas as pd
import pickle

def extract_all_actions():
    """
    Extract all actions from every match of the StatsBomb dataset and stores it
    in a different pickle file for every match.
    """
    competitions_df = extract_events.get_competitions()

    for index, row in competitions_df.iterrows():
        competition_id = row['competition_id']
        season_id = row['season_id']

        games_df = extract_events.get_games(competition_id, season_id)
        for _index, _row in games_df.iterrows():
            match_id = _row['match_id']
            events = extract_events.get_events(match_id)

            actions = to_actions.convert_to_actions(events)
            match_id = int(match_id)
            actions.to_pickle('dataframes/' + str(match_id) + '.pkl')
        print("DONE:", competition_id)


def extract_shots():
    """
    Extract all shots from every match of the StatsBomb dataset and stores it
    in a single pickle file.
    """
    competitions_df = extract_events.get_competitions()

    all_shots = pd.DataFrame()
    for index, row in competitions_df.iterrows():
        competition_id = row['competition_id']
        season_id = row['season_id']

        games_df = extract_events.get_games(competition_id, season_id)
        
        for _index, _row in games_df.iterrows():
            match_id = _row['match_id']
            events = extract_events.get_events(match_id)
            shots_df = shots.get_shots(events)
            frames = [all_shots, shots_df]
            all_shots = pd.concat(frames, ignore_index=True)
        print("DONE:", competition_id)
    all_shots.to_pickle('total_shots.pkl')


if __name__ == '__main__':
    #extract_shots()
    extract_all_actions()
    



    
    
