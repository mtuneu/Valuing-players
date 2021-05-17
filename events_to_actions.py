import pandas as pd
from scipy.spatial import distance
from math import acos, degrees

def convert_to_actions(events):
    """
    Receives a dataframe of events form an specific match and converts it to actions.

    Returns a new dataframe with the actions and the following columns: [
        ['game_id', 'original_event_id', 'team_id', 'period', 'player_id', 'outcome', 'type_id', 
        'type_name', 'body_part', 'situation', 'distance', 'angle', 'prev_type_id']
    ]

    """
    actions = pd.DataFrame()
    events = events.fillna(0)
    goal = (120, 40)

    for index, row in events.iterrows():
        actions.loc[index, 'possession_team'] = row['possession_team']
        actions.loc[index, 'game_id'] = row['match']
        actions.loc[index, 'original_event_id'] = row['id']
        actions.loc[index, 'team_id'] = row['team']
        actions.loc[index, 'period'] = row['period']
        actions.loc[index, 'player_id'] = row['player']
        actions.loc[index, 'outcome'] = 1
        actions.loc[index, 'start_x'] = row.location[0] if row.location else 1
        if actions.loc[index, 'start_x'] == 120.0:
            actions.loc[index, 'start_x'] = 119.0
        actions.loc[index, 'start_y'] = row.location[1] if row.location else 1
        if row['type_name'] == ('pass' or 'carry' or 'shot'):
            actions.loc[index, 'end_x'] = row[row['type_name']]['end_location'][0]
        else:
            actions.loc[index, 'end_x'] = actions.loc[index, 'start_x']
        
        if row['type_name'] == ('pass' or 'carry' or 'shot'):
            actions.loc[index, 'end_y'] = row[row['type_name']]['end_location'][1]
        else:
            actions.loc[index, 'end_y'] = actions.loc[index, 'start_y']

        actions.loc[index, 'type_id'] = row['type_id']
        actions.loc[index, 'type_name'] = " "
        actions.loc[index, 'body_part'] = 1
        actions.loc[index, 'situation'] = 65
        actions.loc[index, 'distance'] = distance.euclidean((actions.loc[index, 'start_x'], actions.loc[index, 'start_y']), goal)
        actions.loc[index, 'vector_1'] = distance.euclidean((actions.loc[index, 'start_x'], actions.loc[index, 'start_y']), (120, 30))
        actions.loc[index, 'vector_2'] = distance.euclidean((actions.loc[index, 'start_x'], actions.loc[index, 'start_y']), (120, 50))
        actions.loc[index, 'vector_3'] = distance.euclidean((120,30) , (120, 50))
        try:
            actions.loc[index, 'angle'] = degrees(acos((actions.loc[index, 'vector_1'] * actions.loc[index, 'vector_1'] + actions.loc[index, 'vector_2'] * actions.loc[index, 'vector_2'] - actions.loc[index, 'vector_3'] * actions.loc[index, 'vector_3'])/(2.0 * actions.loc[index, 'vector_1'] * actions.loc[index, 'vector_2'])))
        except:
            print(actions.loc[index, 'vector_1'], actions.loc[index, 'vector_2'], actions.loc[index, 'vector_3'], actions.loc[index, 'start_x'], actions.loc[index, 'start_y'])
            print("TYPE", actions.loc[index, 'type_id'])
        if row['type_id'] == 33:
            #50/50
            actions.loc[index, 'type_name'] = "Tackle"
            #Problem with normalize 50_50
            actions.loc[index, 'outcome'] = 1
            actions.loc[index, 'type_id'] = 4
        elif row['type_id'] == 30:
            #Pass
            outcome, type_id, type_name = normalize_pass(row[row['type_name']])
            actions.loc[index, 'outcome'] = outcome
            actions.loc[index, 'type_id'] = type_id
            actions.loc[index, 'type_name'] 
        elif row['type_id'] == 4:
            #Duel
            actions.loc[index, 'type_name'] = "Tackle"
            actions.loc[index, 'outcome'] = normalize_duel(row[row['type_name']])
        elif row['type_id'] == 14:
            #Dribble
            actions.loc[index, 'type_name'] = "Dribble"
            actions.loc[index, 'outcome'] = normalize_dribble(row[row['type_name']])
        elif row['type_id'] == 16:
            #Shot
            type_name, outcome, bp, sit = normalize_shot(row[row['type_name']])
            actions.loc[index, 'outcome'] = outcome
            actions.loc[index, 'type_name'] = type_name
            actions.loc[index, 'body_part'] = bp
            actions.loc[index, 'situation'] = sit
        elif row['type_id'] == 10:
            #Interception
            actions.loc[index, 'type_name'] = "Interception"
            actions.loc[index, 'outcome'] = normalize_interception(row[row['type_name']])
        elif row['type_id'] == 22:
            #Foul
            actions.loc[index, 'type_name'] = "Foul"
            actions.loc[index, 'outcome'] = 0
        elif row['type_id'] == 9:
            #Clearance
            actions.loc[index, 'type_name'] = "Clearance"
            actions.loc[index, 'outcome'] = 1
        elif row['type_id'] == 43:
            #Dribble
            actions.loc[index, 'type_name'] = 'Dribble'
            actions.loc[index, 'outcome'] = 1
        elif row['type_id'] == 2:
            #Interception
            actions.loc[index, 'type_name'] = "Interception"
            actions.loc[index, 'outcome'] = 1
            actions.loc[index, 'type_id'] = 10
        elif row['type_id'] == 3 or row['type_id'] == 37 or row['type_id'] == 38:
            #Loses ball
            actions.loc[index, 'type_name'] = "Dispossessed"
            actions.loc[index, 'outcome'] = 0
            actions.loc[index, 'type_id'] = 3 
        elif row['type_id'] == 23:
            #Goalkeeper
            type_id, type_name, outcome = normalize_goalkeeper(row['goalkeeper'])
            actions.loc[index, 'type_id'] = 23
            actions.loc[index, 'type_name'] = type_name
            actions.loc[index, 'outcome'] = outcome
        elif row['type_id'] == 20:
            #Own Goal
            actions.loc[index, 'type_id'] = 20
            actions.loc[index, 'type_name'] = 'own_goal'
            actions.loc[index, 'outcome'] = 0
        actions.loc[index, 'prev_type_id'] = actions.loc[index - 1, 'type_id'] if index > 0 else 30
        actions.loc[index, 'possession_team'] = row['possession_team']

    actions = actions.reset_index()
    actions = actions.drop(columns=["index", 'vector_1', 'vector_2', 'vector_3', 'start_x', 'start_y', 'end_x', 'end_y'])

    return actions


def normalize_50_50(action_dict):
    if 'outcome' in action_dict:
        if action_dict['outcome']['id'] == 108 or action_dict['outcome']['id'] == 147:
            return 1
        else:
            return 0
    else:
        return 0

def normalize_pass(action_dict):
    if 'outcome' in action_dict:
        outcome = 0
    else:
        outcome = 1
    
    if 'cross' in action_dict:
        type_id = 32
        type_name = 'cross'
    elif 'cut-back' in action_dict:
        type_id = 33
        type_name = 'cut-back'
    elif 'type' in action_dict:
        if action_dict['type']['id'] == 61:
            type_id = 61
            type_name = 'corner'
        elif action_dict['type']['id'] == 62:
            type_id = 62
            type_name = 'free kick'
        else:
            type_id = 30
            type_name = 'pass'
    else:
        type_id = 30
        type_name = 'pass'

    return outcome, type_id, type_name

def normalize_shot(action_dict):
    if 'outcome' in action_dict:
        if action_dict['outcome']['id'] == 97:
            outcome = 1
        else:
            outcome = 0
    else:
        outcome = 1
    
    type_name = 'Shot'
    
    if action_dict['body_part']['id'] == 37 or action_dict['body_part']['id'] == 70:
        bp = 2
    else:
        bp = 1
    
    if action_dict['type']['id'] == 61:
        #Corner
        situation = 61
    elif action_dict['type']['id'] == 62:
        #Free kick
        situation = 62
    elif action_dict['type']['id'] == 87 or action_dict['type']['id'] == 65:
        #Open play
        situation = 65
    else:
        #Penalty
        situation = 63


    return type_name, outcome, bp, situation


def normalize_interception(action_dict):
    if action_dict['outcome'] == 1 or action_dict['outcome'] == 13 or action_dict['outcome'] == 14:
        return 0
    else:
        return 1

def normalize_duel(action_dict):
    if 'outcome' in action_dict:
        if action_dict['outcome']['id'] == 1 or action_dict['outcome']['id'] == 13 or action_dict['outcome']['id'] == 14:
            return 0
    else:
        return 1

def normalize_dribble(action_dict):
    if action_dict['outcome']['id'] == 9:
        return 0
    else:
        return 1

def normalize_goalkeeper(action_dict):
    if action_dict['type']['id'] != 26 or action_dict['type']['id'] != 28 or action_dict['type']['id'] != 32:
        if 'outcome' in action_dict:
            if action_dict['outcome']['id'] == 50 or action_dict['outcome']['id'] == 55:
                outcome = 0
            else:
                outcome = 1
        else:
            outcome = 1
    else:
        outcome = 0
    
    return 23, "Save", outcome
