import pandas as pd
from scipy.spatial import distance
from math import acos, degrees
import warnings

warnings.filterwarnings('ignore')

def get_shots(events_df):
    """
    Returns a dataframe with the actions form with all the shots.
    """
    events_df = events_df[events_df.type_id != 42]
    events_df = events_df.reset_index(drop=True)
    shots_df = events_df[events_df['type_id'] == 16]
    aux_shots = events_df.loc[shots_df.index - 1]
    shots_df['distance'] = shots_df.apply(lambda row: distance.euclidean( (row.location[0], row.location[1]) , (120, 40) ), axis=1)
    shots_df['vector_1'] = shots_df.apply(lambda row: distance.euclidean( (row.location[0], row.location[1]) , (120, 30) ) ,axis=1)
    shots_df['vector_2'] = shots_df.apply(lambda row: distance.euclidean( (row.location[0], row.location[1]) , (120, 50) ) ,axis=1)
    shots_df['vector_3'] = shots_df.apply(lambda row: distance.euclidean( (120,30) , (120, 50)), axis=1)
    shots_df['angle'] = shots_df.apply(lambda row: degrees(acos((row.vector_1 * row.vector_1 + row.vector_2 * row.vector_2 - row.vector_3 * row.vector_3)/(2.0 * row.vector_1 * row.vector_2))) , axis=1)
    frames = [shots_df, aux_shots]
    shots_df = pd.concat(frames)
    shots_df = shots_df.sort_values(by=['index'])
    for index, row in shots_df.iterrows():
        try:
            if shots_df.loc[index, 'type_id'] == 16:
                shots_df.loc[index, 'start_x'] = row.location[0]
                shots_df.loc[index, 'start_y'] = row.location[1]
                shots_df.loc[index, 'prev_type_id'] = shots_df.loc[index - 1, 'type_id']
                shots_df.loc[index, 'outcome'] = 1 if shots_df.loc[index, 'shot']['outcome']['id'] == 97 else 0
                if 'statsbomb_xg' in shots_df.loc[index, 'shot']:
                    shots_df.loc[index, 'statsbomb_xg'] = shots_df.loc[index, 'shot']['statsbomb_xg']
                if 'body_part' in shots_df.loc[index, 'shot']:
                    aux_dict = shots_df.loc[index, 'shot']
                    shots_df.loc[index, 'body_part'] = 37 if aux_dict['body_part']['id'] == 37 else 38
                if 'type' in shots_df.loc[index, 'shot']:
                    situation = shots_df.loc[index, 'shot']['type']['id']
                    if situation == 61:
                        shots_df.loc[index, 'situation'] = situation
                    elif situation == 62:
                        shots_df.loc[index, 'situation'] = situation
                    elif situation == 68:
                        shots_df.loc[index, 'situation'] = situation
                    else:
                        shots_df.loc[index, 'situation'] = 87
            elif shots_df.loc[index, 'type_id'] == 30:
                if 'cross' in shots_df.loc[index, 'pass']:
                    shots_df.loc[index, 'type_id'] = 200
                elif 'cut-back' in shots_df.loc[index, 'pass']:
                    shots_df.loc[index, 'type_id'] = 201
                elif 'type' in shots_df.loc[index, 'pass']:
                    if shots_df.loc[index, 'pass']['type']['id'] == 61 or shots_df.loc[index, 'pass']['type']['id'] == 62:
                        shots_df.loc[index, 'type_id'] = 200
        except:
            print(shots_df.loc[index, 'type_id'])
    shots_df = shots_df.dropna(how='any', subset=['prev_type_id'])
    shots_df = shots_df[['type_id', 'prev_type_id', 'situation', 'distance', 'body_part', 'angle', 'statsbomb_xg','outcome', 'start_x', 'start_y']].copy()
    
                
    return shots_df
