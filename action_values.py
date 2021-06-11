from events_extraction.extract_events import get_competitions, get_games
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

def load_match_events(match_id):
    match_df = pd.read_pickle('dataframes/'+str(match_id)+'.pkl')

    return match_df

def get_all_values(xg_model):
    competitions_df = get_competitions()

    for index, row in competitions_df.iterrows():
        competition_id = row['competition_id']
        season_id = row['season_id']

        games_df = get_games(competition_id, season_id)

        for _index, _row in games_df.iterrows():
            match_id = _row['match_id']

            get_one_match_values(xg_model, match_id)

def get_one_match_values(xg_model, match_id):
    match_df = load_match_events(match_id)

    team_id = match_df.loc[0, 'team_id']

    score_df = calculate_scoring_probabilities(match_df, team_id, xg_model)

    concede_df = calculate_conceding_probabilities(match_df, team_id, xg_model)

    value_df = score_df.merge(concede_df)

    simple_values_df = get_sequence(value_df, team_id)

    values_df = get_total_value(simple_values_df)

    values_df.to_pickle('match_values/'+str(match_id)+'.pkl')

    return values_df


if __name__ == '__main__':

    xg_model = keras.models.load_model('models/nn_model_v2.h5')

    get_all_values(xg_model)