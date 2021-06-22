from events_extraction import extract_events
from events_to_actions import events_to_actions as to_actions
import argparse
from events_to_actions import shots
import pandas as pd

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
    all_shots.to_pickle('dataframes/total_shots.pkl')


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='All events or shots')
    my_parser.add_argument('Type', metavar='type', type=str, help='all events or shots')
    args = my_parser.parse_args()

    types = args.Type

    if types == 'shots':
        print("Doing shots")
        extract_shots()
    else:
        print("Doing all events")
        extract_all_actions()
    



    
    
