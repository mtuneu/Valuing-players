import pandas as pd
import warnings
import json

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def get_competitions():
    """
    Read all the competitions from the StatsBomb dataset.

    Returns a dataframe with columns: [
        'competition_id' , 'season_id', 'country_name', 'competition_name', 
        'competition_gender', and 'season_name'
    ]
    """
    with open('./data/competitions.json', encoding='utf-8') as json_file:
        data = json.load(json_file)

    competition_df = pd.DataFrame(data).drop(
        columns=['match_updated', 'match_available'])

    return competition_df


def get_games(competition_id, season_id):
    """
    Read all games from a specific competition and season.

    Returns a dataframe with the columns: [
        'match_id', 'match_date', 'competition_id', 'season_id', 'home_team_id',
        'away_team_id', 'home_score', 'away_score', 'match_week', 'competition_stage_name', 
        'stadium_id', 'stadium_name', 'stadium_extra', 'referee_id'
    ]
    """
    with open('./data/matches/'+str(competition_id)+'/'+str(season_id)+'.json',
              encoding='utf-8') as json_file:
        data = json.load(json_file)

    games_df = pd.DataFrame(data)

    return games_df


def get_teams(match_id):
    """
    Read the teams playing on a specific match.

    Returns a list with the id of the teams that are playing an specific match.
    """

    with open('./data/lineups/' + str(match_id) + '.json', encoding='utf-8') as json_file:
        data = json.load(json_file)

    lineups_df = pd.DataFrame(data)

    teams = []

    for index, row in lineups_df.iterrows():
        teams.append(row['team_id'])

    return teams


def get_events(match_id):
    """
    Get all the events for an specific match.

    Returns a DataFrame with the columns: [
        ['id', 'index', 'period', 'timestamp', 'minute', 'second', 
        'possession', 'possession_team', 'play_pattern', 'team', 'duration', 
        'player', 'position', 'location', 'pass', 'carry', 'ball_receipt', 'duel', 
        'foul_committed', 'shot', 'goalkeeper', 'clearance', 'foul_won', 
        'interception', 'ball_recovery', 'counterpress', 
        'out', 'dribble', '50_50', 'substitution', 'block', 'type_id', 'type_name', 'match']
    ]


    """

    with open('./data/events/' + str(match_id) + '.json', encoding='utf-8') as json_file:
        data = json.load(json_file)
    events_df = pd.DataFrame(data)
    try:
        events_df = events_df.drop(columns=['under_pressure'], axis=1)
    except:
        pass
    try:
        events_df = events_df.drop(columns=['off_camera'], axis=1)
    except:
        pass
    try:
        events_df = events_df.drop(columns=['related_events'], axis=1)
    except:
        pass
    try:
        events_df = events_df.drop(columns=['tactics'], axis=1)
    except:
        pass
    events_df = events_df[events_df['type'] != {'id': 18, 'name' : 'Half Start'}]
    events_df = events_df[events_df['type'] != {'id': 35, 'name' : 'Starting XI'}]
    events_df = events_df[events_df['type'] != {'id': 40, 'name' : 'Injury Stoppage'}]
    events_df = events_df[events_df['type'] != {'id': 39, 'name' : 'Dribbled Past'}]
    events_df = events_df[events_df['type'] != {'id': 41, 'name' : 'Referee Ball-Drop'}]
    events_df = events_df[events_df['type'] != {'id': 36, 'name' : 'Tactical Shift'}]
    events_df = events_df[events_df['type'] != {'id': 34, 'name' : 'Half End'}]
    events_df = events_df[events_df['type'] != {'id': 26, 'name' : 'Player On'}]
    events_df = events_df[events_df['type'] != {'id': 27, 'name' : 'Player Off'}]
    events_df = events_df[events_df['type'] != {'id': 19, 'name' : 'Substitution'}]
    events_df = events_df[events_df['type'] != {'id': 8, 'name' : 'Offside'}]
    events_df = events_df[events_df['type'] != {'id': 5, 'name' : 'Camera On'}]
    events_df = events_df[events_df['type'] != {'id': 42, 'name' : 'Ball Receipt*'}]
    events_df = events_df[events_df['type'] != {'id': 24, 'name' : 'Bad Behaviour'}]


    events_df = events_df.reset_index(drop=True)

    for index in range(len(events_df)):
        events_df.loc[index, 'type_id'] = events_df.loc[index, 'type'].get('id')
        events_df.loc[index, 'type_name'] = events_df.loc[index, 'type'].get('name').lower()
        events_df.loc[index, 'possession_team'] = events_df.loc[index, 'possession_team'].get('id')
        events_df.loc[index, 'play_pattern'] = events_df.loc[index, 'play_pattern'].get('id')
        events_df.loc[index, 'team'] = events_df.loc[index, 'team'].get('id')
        if(type(events_df.loc[index, 'player']) == dict):
            events_df.loc[index, 'player'] = events_df.loc[index, 'player'].get('id')
        if(type(events_df.loc[index, 'position']) == dict):
            events_df.loc[index, 'position'] = events_df.loc[index, 'position'].get('id')

    events_df['match'] = match_id

    events_df = events_df.drop(columns=['type'])

    return events_df
