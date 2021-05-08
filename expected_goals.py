import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, f1_score, precision_score, recall_score, plot_roc_curve,r2_score, mean_squared_error


def read_actions():
    shots = pd.read_pickle('total_shots.pkl')
    
    return shots

def find_best_hyperparameters(shots):
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]

    columns_features = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']
    statsbomb_xg = ['statsbomb_xg']
    columns_target = 'outcome'

    X = shots[columns_features]
    y = shots[columns_target]

    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X,y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param)) 

    #Best liblinear - penalty: l2 - C:10
    
def logistic_expected_goals(shots, solver, cw):

    columns_features = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']
    statsbomb_xg = ['statsbomb_xg']
    columns_target = 'outcome'

    X = shots[columns_features]
    y = shots[columns_target]
    c = shots[statsbomb_xg]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    S_train, S_test, c_train, c_test = train_test_split(X, c, test_size=0.2)

    #model = LogisticRegression(max_iter=6000, class_weight=cw,solver=solver, C=10)
    #model.fit(X_train, y_train)

    model = load('models/logistic_expected_goals.joblib')

    score = model.score(X=X_test,y=y_test)
    roc = roc_auc_score(y_test,model.predict_proba(X_test)[:,1])*100
    
    print('The accuracy of classifying whether a shot is goal or not is {}%'.format(score*100))
    print('Our classifier obtains a ROC score of {}%'.format(round(roc,2)))
    print('Our classifier obtains an R2 score of {}'.format(round(r2_score(y_test,model.predict_proba(X_test)[:,1]),4)))
    print('Our classifier obtains an MSE score of {}'.format(round(mean_squared_error(c_test, model.predict_proba(X_test)[:,1]),4)))

    dump(model, 'logistic_expected_goals.joblib')

    return score

def logistic_expected_goals_v2(shots, solver, cw):

    columns_features = ['distance', 'angle', 'prev_type_id', 'body_part', 'situation']
    statsbomb_xg = ['statsbomb_xg']
    columns_target = 'outcome'

    X = shots[columns_features]
    y = shots[columns_target]
    c = shots[statsbomb_xg]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    S_train, S_test, c_train, c_test = train_test_split(X, c, test_size=0.2)

    model = LogisticRegression(max_iter=6000, class_weight=cw,solver=solver, C=10)
    model.fit(X_train, y_train)

    score = model.score(X=X_test,y=y_test)
    roc = roc_auc_score(y_test,model.predict_proba(X_test)[:,1])*100

    print('The accuracy of classifying whether a shot is goal or not is {}%'.format(score*100))
    print('Our classifier obtains a ROC score of {}%'.format(round(roc,2)))
    print('Our classifier obtains an R2 score of {}'.format(round(r2_score(y_test,model.predict_proba(X_test)[:,1]),4)))
    print('Our classifier obtains an MSE score of {}'.format(round(mean_squared_error(c_test, model.predict_proba(X_test)[:,1]),4)))

    return score
 
if __name__ == "__main__":
    shots = read_actions()
    print("FIRST MODEL:")
    logistic_expected_goals(shots, 'lbfgs', {0:3, 1:2.5})
    #print("SECOND MODEL:")
    #logistic_expected_goals_v2(shots, 'liblinear', {0:3, 1:2.5})