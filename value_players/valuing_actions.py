
import pandas as pd
from scipy.spatial import distance
from math import acos, degrees


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

def return_score_probabilites(team_id, row):
    if row['team_id'] == team_id:
        return row['score_prob']
    else:
        return row['concede_prob']

def return_concede_probabilites(team_id, row):
    if row['team_id'] == team_id:
        return row['concede_prob']
    else:
        return row['score_prob']



def get_sequence(match_df, team_id):

    match_df['attacking_value'] = 0.0
    match_df['deffensive_value'] = 0.0
    team1_xG = 0
    team2_xG = 0
    team_1_goals = 0
    team_2_goals = 0
    for index, row in match_df.iterrows():
        try:
            if (index < (len(match_df.index) - 2)):
                if(row['type_id'] == 16 and row['outcome'] == 1):
                    if row['team_id'] == team_id:
                        team_1_goals+=1
                        team1_xG = match_df.loc[index, 'score_prob'] + team1_xG
                        match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'score_prob']
                        match_df.at[index, 'deffensive_value'] = 1 - match_df.loc[index, 'concede_prob']
                    else:
                        team_2_goals+=1
                        team2_xG = match_df.at[index, 'concede_prob'] + team2_xG
                        match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'concede_prob']
                        match_df.at[index, 'deffensive_value'] = 1 - match_df.loc[index, 'score_prob']
                else:    
                    if row['team_id'] == team_id:
                        if row['type_id'] == 16:
                            team1_xG = match_df.loc[index, 'score_prob'] + team1_xG
                        match_df.at[index, 'attacking_value'] = ((return_score_probabilites(team_id, match_df.loc[index + 1]) - match_df.at[index , 'score_prob']) + (return_score_probabilites(team_id, match_df.loc[index+2]) - return_score_probabilites(team_id, match_df.loc[index + 1]))) / 2
                        match_df.at[index, 'deffensive_value'] = - (((return_concede_probabilites(team_id, match_df.loc[index + 1]) - match_df.at[index , 'concede_prob']) + (return_concede_probabilites(team_id, match_df.loc[index + 2]) - return_concede_probabilites(team_id, match_df.loc[index + 1]))) / 2)
                    else:
                        if row['type_id'] == 16:
                            team2_xG =match_df.loc[index, 'concede_prob'] + team2_xG
                        match_df.at[index, 'deffensive_value'] = - (((return_score_probabilites(team_id, match_df.loc[index + 1]) - match_df.at[index , 'score_prob']) + (return_score_probabilites(team_id, match_df.loc[index+2]) - return_score_probabilites(team_id, match_df.loc[index + 1]))) / 2)
                        match_df.at[index, 'attacking_value'] = ((return_concede_probabilites(team_id, match_df.loc[index + 1]) - match_df.at[index , 'concede_prob']) + (return_concede_probabilites(team_id, match_df.loc[index + 2]) - return_concede_probabilites(team_id, match_df.loc[index + 1]))) / 2
            elif (index < len(match_df.index) -1):
                if(row['type_id'] == 16 and row['outcome'] == 1):
                    if row['team_id'] == team_id:
                        team_1_goals+=1
                        team1_xG = match_df.loc[index, 'score_prob'] + team1_xG
                        match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'score_prob']
                        match_df.at[index, 'deffensive_value'] = 1 - match_df.loc[index, 'concede_prob']
                    else:
                        team_2_goals+=1
                        team2_xG = match_df.at[index, 'concede_prob'] + team2_xG
                        match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'concede_prob']
                        match_df.at[index, 'deffensive_value'] = 1 - match_df.loc[index, 'score_prob']
                else:
                    if row['team_id'] == team_id:
                        if row['type_id'] == 16:
                            team1_xG = match_df.loc[index, 'concede_prob'] + team1_xG
                        match_df.at[index, 'attacking_value'] = (return_score_probabilites(team_id, match_df.loc[index + 1]) - match_df.at[index, 'score_prob'])
                        match_df.at[index, 'deffensive_value'] = -(match_df.at[index + 1, 'concede_prob'] - match_df.at[index, 'concede_prob'])
                    else:
                        if row['type_id'] == 16:
                            team2_xG = match_df.loc[index, 'concede_prob'] + team2_xG
                        match_df.at[index, 'deffensive_value'] = - (return_score_probabilites(team_id, match_df.loc[index + 1]) - match_df.at[index, 'score_prob'])
                        match_df.at[index, 'attacking_value'] = (return_concede_probabilites(team_id, match_df.loc[index + 1]) - match_df.at[index, 'concede_prob'])
            else:
                if(row['type_id'] == 16 and row['outcome'] == 1):
                    if row['team_id'] == team_id:
                        team_1_goals+=1
                        team1_xG = match_df.loc[index, 'score_prob'] + team1_xG
                        match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'score_prob']
                        match_df.at[index, 'deffensive_value'] = 1 - match_df.loc[index, 'concede_prob']
                    else:
                        team_2_goals+=1
                        team2_xG = match_df.at[index, 'concede_prob'] + team2_xG
                        match_df.at[index, 'attacking_value'] = 1 - match_df.loc[index, 'concede_prob']
                        match_df.at[index, 'deffensive_value'] = 1 - match_df.loc[index, 'score_prob']
                else:
                    if row['team_id'] == team_id:
                        if row['type_id'] == 16:
                            team1_xG = match_df.at[index, 'concede_prob'] + team1_xG
                        match_df.at[index, 'attacking_value'] = match_df.at[index, 'score_prob']
                        match_df.at[index, 'deffensive_value'] = - match_df.at[index, 'concede_prob']
                    else:
                        if row['type_id'] == 16:
                            team2_xG = match_df.at[index, 'concede_prob'] + team2_xG
                        match_df.at[index, 'deffensive_value'] = - match_df.at[index, 'score_prob']
                        match_df.at[index, 'attacking_value'] = match_df.at[index, 'concede_prob']
        except:
            print("ERROR")

        match_df['team1_xG'] = team1_xG
        match_df['team2_xG'] = team2_xG
        match_df['team1_goals'] = team_1_goals
        match_df['team2_goals'] = team_2_goals
    return match_df

def get_total_value(values_df):

    values_df['total_value'] = values_df['attacking_value'] + values_df['deffensive_value']

    return values_df


    