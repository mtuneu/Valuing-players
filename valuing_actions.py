from extract_events import get_competitions, get_games
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from joblib import load
from tensorflow import keras
from scipy.spatial import distance
from math import acos, degrees

def load_match_events(match_id):
    match_df = pd.read_pickle('dataframes/'+str(match_id)+'.pkl')

    return match_df

def calculate_scoring_probabilities(match_df, team_id,xg_model):
    columns = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']
    o_goal = (0, 40)

    for index, row in match_df.iterrows():
        if team_id != match_df.loc[index, 'team_id']:
            match_df.loc[index, 'distance'] = distance.euclidean((match_df.loc[index, 'start_x'], match_df.loc[index, 'start_y']), o_goal)
            match_df.loc[index, 'vector_1'] = distance.euclidean((match_df.loc[index, 'start_x'], match_df.loc[index, 'start_y']), (0, 30))
            match_df.loc[index, 'vector_2'] = distance.euclidean((match_df.loc[index, 'start_x'], match_df.loc[index, 'start_y']), (0, 50))
            match_df.loc[index, 'vector_3'] = distance.euclidean((0,30) , (0, 50))
            match_df.loc[index, 'angle'] = degrees(acos((match_df.loc[index, 'vector_1'] * match_df.loc[index, 'vector_1'] + match_df.loc[index, 'vector_2'] * match_df.loc[index, 'vector_2'] - match_df.loc[index, 'vector_3'] * match_df.loc[index, 'vector_3'])/(2.0 * match_df.loc[index, 'vector_1'] * match_df.loc[index, 'vector_2'])))


    proba_df = pd.DataFrame(match_df[['team_id', 'possession_team', 'original_event_id', 'player_id', 'outcome', 'type_id']])
    match_df = match_df[columns]

    

    p_a = xg_model.predict(match_df)

    proba_df['score_prob'] = p_a

    return proba_df

def calculate_conceding_probabilities(match_df, team_id, xg_model):
    columns = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']

    o_goal = (0, 40)

    for index, row in match_df.iterrows():
        if team_id == match_df.loc[index, 'team_id']:
            match_df.loc[index, 'distance'] = distance.euclidean((match_df.loc[index, 'start_x'], match_df.loc[index, 'start_y']), o_goal)
            match_df.loc[index, 'vector_1'] = distance.euclidean((match_df.loc[index, 'start_x'], match_df.loc[index, 'start_y']), (0, 30))
            match_df.loc[index, 'vector_2'] = distance.euclidean((match_df.loc[index, 'start_x'], match_df.loc[index, 'start_y']), (0, 50))
            match_df.loc[index, 'vector_3'] = distance.euclidean((0,30) , (0, 50))
            match_df.loc[index, 'angle'] = degrees(acos((match_df.loc[index, 'vector_1'] * match_df.loc[index, 'vector_1'] + match_df.loc[index, 'vector_2'] * match_df.loc[index, 'vector_2'] - match_df.loc[index, 'vector_3'] * match_df.loc[index, 'vector_3'])/(2.0 * match_df.loc[index, 'vector_1'] * match_df.loc[index, 'vector_2'])))

    proba_df = pd.DataFrame(match_df[['team_id', 'possession_team', 'original_event_id', 'player_id', 'outcome', 'type_id']])
    match_df = match_df[columns]

    
    p_a = xg_model.predict(match_df)

    proba_df['concede_prob'] = p_a

    return proba_df



def get_sequence(match_df):

    match_df['attacking_value'] = 0.0
    match_df['deffensive_value'] = 0.0

    for index, row in match_df.iterrows():
        try:
            if (index < (len(match_df.index) - 2)):
                if(row['type_id'] == 16 and row['outcome'] == 1):
                    match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'score_prob']
                    match_df.at[index, 'deffensive_value'] = 1 - match_df.at[index, 'concede_prob']
                else:
                    match_df.at[index, 'attacking_value'] = (match_df.at[index, 'score_prob'] + match_df.at[index + 1, 'score_prob'] + match_df.at[index + 2, 'score_prob']) / 3
                    match_df.at[index, 'deffensive_value'] = (match_df.at[index, 'concede_prob'] + match_df.at[index + 1, 'concede_prob'] + match_df.at[index + 2, 'concede_prob']) / 3
            elif (index < len(match_df.index) -1):
                if(row['type_id'] == 16 and row['outcome'] == 1):
                    match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'score_prob']
                    match_df.at[index, 'deffensive_value'] = 1 - match_df.loc[index, 'concede_prob']
                else:
                    match_df.at[index, 'attacking_value'] = (match_df.at[index, 'score_prob'] + match_df.at[index + 1, 'score_prob']) / 2
                    match_df.at[index, 'deffensive_value'] = (match_df.at[index, 'concede_prob'] + match_df.at[index + 1, 'concede_prob']) / 2
            else:
                if(row['type_id'] == 16 and row['outcome'] == 1):
                    match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'score_prob']
                    match_df.at[index, 'deffensive_value'] = 1 - match_df.loc[index, 'concede_prob']
                else:
                    match_df.at[index, 'attacking_value'] = match_df.at[index, 'score_prob']
                    match_df.at[index, 'deffensive_value'] = match_df.at[index, 'concede_prob']
        except:
            print("ERROR")
    
    return match_df

def get_total_value(values_df):

    values_df['total_value'] = values_df['attacking_value'] - values_df['deffensive_value']

    return values_df

def get_all_values(xg_model):
    competitions_df = get_competitions()

    for index, row in competitions_df.iterrows():
        competition_id = row['competition_id']
        season_id = row['season_id']

        games_df = get_games(competition_id, season_id)

        for _index, _row in games_df.iterrows():
            match_id = _row['match_id']

            match_df = load_match_events(match_id)

            team_id = match_df.loc[0, 'team_id']

            score_df = calculate_scoring_probabilities(match_df, team_id, xg_model)

            concede_df = calculate_conceding_probabilities(match_df, team_id, xg_model)

            value_df = score_df.merge(concede_df)

            simple_values_df = get_sequence(value_df)

            values_df = get_total_value(simple_values_df)

            values_df.to_pickle('match_values/'+str(match_id)+'.pkl')
        
        print("DONE")

def get_one_match_values(xg_model, match_id):
    match_df = load_match_events(match_id)

    team_id = match_df.loc[0, 'team_id']

    score_df = calculate_scoring_probabilities(match_df, team_id, xg_model)

    concede_df = calculate_conceding_probabilities(match_df, team_id, xg_model)

    value_df = score_df.merge(concede_df)

    simple_values_df = get_sequence(value_df)

    values_df = get_total_value(simple_values_df)

    values_df.to_pickle('match_values/'+str(match_id)+'.pkl')


if __name__ == '__main__':

    xg_model = keras.models.load_model('models/nn_model_v2.h5')

    match_id = 7430

    get_one_match_values(xg_model, match_id)

    #get_all_values(xg_model)

    