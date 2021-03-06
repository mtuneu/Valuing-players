import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers
from xg_models.expected_goals import read_actions
from sklearn.model_selection import train_test_split


def build_model(hp):
    """
    Builds and saves the neural network model
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(4,)))
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='binary_crossentropy', metrics=['accuracy'])
    model.save('../models/nn_model_v2.h5')
    return model

def cnn(shots):
    """
    Loads, fits and evalutes the neural network model
    """
    #columns_features = ['start_distance', 'body_part', 'prev_start_distance', 'angle', 'situation', 'prev_type_id']
    columns_features = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']
    columns_target = 'outcome'

    X = shots[columns_features]
    y = shots[columns_target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    c = shots['statsbomb_xg']

    model = keras.models.load_model('../models/nn_model_v2.h5')
    
    history = model.fit(x_train, y_train, epochs=20)
    test_scores = model.evaluate(x_test, y_test, verbose=2)


    return test_scores

if __name__ == "__main__":
    shots = read_actions()

    test_scores = cnn(shots)