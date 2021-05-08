import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import expected_goals as xG
import nn_expected_goals as nn_xg
from expected_goals import read_actions
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from joblib import dump, load
from tensorflow import keras
import numpy as np
import warnings

warnings.filterwarnings("ignore")


LOGISTIC_PATH = 'models/logistic_expected_goals.joblib'
NN_PATH = 'models/nn_model_v2.h5'



def compare_classes(file_path_1, file_path_2, shots):
    columns_features = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']
    columns_target = 'outcome'

    X = shots[columns_features]
    y = shots[columns_target]

    log_model = load(file_path_1)
    nn_model = keras.models.load_model(file_path_2)

    log_accuracies = []
    log_precisions = []
    nn_accuracies = []
    nn_precisions = []

    accuracies = []
    precisions = []


    i = 0

    while i < 10:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        log_predicted = log_model.predict(x_test)
        nn_predicted = nn_model.predict_classes(x_test)

        log_acc = metrics.accuracy_score(y_test, log_predicted)
        log_pr = metrics.precision_score(y_test, log_predicted)
        nn_acc = metrics.accuracy_score(y_test, nn_predicted)
        nn_pr = metrics.precision_score(y_test, nn_predicted)


        log_accuracies.append(log_acc)
        nn_accuracies.append(nn_acc)
        log_precisions.append(log_pr)
        nn_precisions.append(nn_pr)


        i += 1

    log_accuracy = sum(log_accuracies) / len(log_accuracies)
    nn_accuracy = sum(nn_accuracies) / len(nn_accuracies)
    log_precision = sum(log_precisions) / len(log_precisions)
    nn_precision = sum(nn_precisions) / len(nn_precisions)


    accuracies.append(log_accuracy)
    accuracies.append(nn_accuracy)
    precisions.append(log_precision)
    precisions.append(nn_precision)


    print("ACCURACIES: ", accuracies)
    print("PRECISIONS: ", precisions)


def compare_probabilities(file_path_1, file_path_2, shots):
    columns_features = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']
    columns_target = 'outcome'

    X = shots[columns_features]
    y = shots[columns_target]

    log_model = load(file_path_1)
    nn_model = keras.models.load_model(file_path_2)

    log_losses_log = []
    log_losses_nn = []
    roc_auc_scores_log = []
    roc_auc_scores_nn = []

    log_losses = []
    roc_auc_scores = []

    i = 0

    while i < 10:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        log_predicted = log_model.predict_proba(x_test)
        nn_predicted = nn_model.predict(x_test)

        lss_log = metrics.log_loss(y_test, log_predicted)
        lss_nn = metrics.log_loss(y_test, nn_predicted)
        r_a_log = metrics.roc_auc_score(y_test, log_predicted[:, 1])
        r_a_nn = metrics.roc_auc_score(y_test, nn_predicted)

        log_losses_log.append(lss_log)
        log_losses_nn.append(lss_nn)
        roc_auc_scores_log.append(r_a_log)
        roc_auc_scores_nn.append(r_a_nn)

        i += 1
    
    log_loss_log = sum(log_losses_log) / len(log_losses_log)
    log_loss_nn = sum(log_losses_nn) / len(log_losses_nn)
    roc_auc_log = sum(roc_auc_scores_log) / len(roc_auc_scores_log)
    roc_auc_nn = sum(roc_auc_scores_nn) / len(roc_auc_scores_nn)

    log_losses.append(log_loss_log)
    log_losses.append(log_loss_nn)
    roc_auc_scores.append(roc_auc_log)
    roc_auc_scores.append(roc_auc_nn)
    
    print("LOG LOSSES: ", log_losses)
    print("ROC AUC SCORES: ", roc_auc_scores)

if __name__ == "__main__":
    shots = read_actions()
    #compare_classes(LOGISTIC_PATH, NN_PATH, shots)
    compare_probabilities(LOGISTIC_PATH, NN_PATH, shots)

