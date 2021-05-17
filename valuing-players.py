import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from joblib import load
from expected_goals import read_actions
from tensorflow import keras

def load_match_events(match_id):
    match_df = pd.read_pickle('dataframes/'+str(match_id)+'.pkl')

    return match_df


def calculate_probabilities(match_df, xg_model):

    columns = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']

    proba_df = pd.DataFrame(match_df[['team_id', 'possession_team', 'original_event_id', 'player_id', 'outcome', 'type_id']])
    match_df = match_df[columns]
    
    p_a = xg_model.predict(match_df)

    proba_df['simple_prob'] = p_a

    return proba_df


def get_sequence(match_df):

    match_df['attacking_value'] = np.nan
    match_df['deffensive_value'] = np.nan


    for index, row in match_df.iterrows():
        try:
            if(row['type_id'] == 16 and row['outcome'] == 1):
                match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'simple_prob']
                match_df.at[index, 'deffensive_value'] = -(match_df.loc[index, 'attacking_value'])
            else:
                if(row['possession_team'] == match_df.loc[index + 1, 'possession_team']):
                    match_df.at[index, 'attacking_value'] = match_df.loc[index + 1, 'simple_prob'] - row['simple_prob']
                    match_df.at[index, 'deffensive_value'] = -(match_df.loc[index, 'attacking_value'])
                else:
                    match_df.at[index, 'attacking_value'] =  - (match_df.at[index, 'deffensive_value'])
                    match_df.at[index, 'deffensive_value'] = - (match_df.at[index, 'attacking_value'])
        except:
            pass
    
    return match_df




if __name__ == '__main__':

    xg_model = keras.models.load_model('models/nn_model_v2.h5')


    match_df = load_match_events(7430)
    
    proba_df = calculate_probabilities(match_df, xg_model)
    
    simple_values_df = get_sequence(proba_df)

    print(simple_values_df)
    