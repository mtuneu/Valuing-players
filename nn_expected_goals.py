import pandas as pd
import extract_events
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from kerastuner.tuners import RandomSearch
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from expected_goals import read_actions
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc ,roc_auc_score, r2_score, mean_squared_error





def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(4,)))
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='binary_crossentropy', metrics=['accuracy'])

    return model




def cnn(shots):
    #columns_features = ['start_distance', 'body_part', 'prev_start_distance', 'angle', 'situation', 'prev_type_id']
    columns_features = ['distance', 'angle', 'prev_type_id', 'body_part']
    columns_target = 'outcome'

    X = shots[columns_features]
    y = shots[columns_target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    c = shots['statsbomb_xg']
    S_train, S_test, c_train, c_test = train_test_split(X, c, test_size=0.2)

    model = keras.models.load_model('nn_model.h5')
    
    history = model.fit(x_train, y_train, epochs=20)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print(model.summary())
    print("Test accuracy: ", test_scores[1])

if __name__ == "__main__":
    shots = read_actions()

    columns_features = ['distance', 'angle', 'prev_type_id', 'body_part']
    columns_target = 'outcome'

    X = shots[columns_features]
    y = shots[columns_target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tuner = RandomSearch(build_model,objective='val_accuracy', max_trials=5, executions_per_trial=3, directory='tuner', project_name='analysis_project_v2')
    tuner.search(x_train,y_train, epochs=20, validation_data=(x_test, y_test))

    model = tuner.get_best_models(num_models=1)

    model[0].save('nn_model_v2.h5')