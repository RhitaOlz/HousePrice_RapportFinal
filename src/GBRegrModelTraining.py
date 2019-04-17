import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Modelling Algorithms
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, GridSearchCV
import lightgbm as lgbm
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score

from ModelsTraining import make_dir, distribution_Plots, trainingDeviance_Plot

def modelTraining(model, model_name, features_name, train_X, train_y, valid_X, valid_y, test_df, results_path):
    # model train
    start = time.time()
    model.fit(train_X, train_y)
    end = time.time()

    # Prediction
    price_pred = model.predict(test_df)
    # Price value save
    if log is True:
        price_pred_df = pd.DataFrame(price_pred)
        price_pred_df.to_csv(results_path + '/' + model_name + '_price_pred_log.csv')
        price_pred = [np.round(i, 9) for i in np.expm1(price_pred)]

    price_pred_df = pd.DataFrame(price_pred)
    price_pred_df.to_csv(results_path + '/' + model_name +'_price_pred_dollars.csv')

    # Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting
    print('\n****** ' + model_name + ' *********')
    print("Train score: {:.4f}".format(model.score(train_X, train_y)))
    print("Validation score: {:.4f}".format(model.score(valid_X, valid_y)))
    print("All training data score: {:.4f}".format(model.score(train_valid_X, train_valid_y)))

    return price_pred_df, model.score(train_X, train_y), model.score(valid_X, valid_y)

if __name__ == '__main__':

    # ------------ Results folders ---------------------------------
    results_path = '../resultsGraphs_GBReg'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    make_dir(results_path)

    # ------------ Load Data --------------------------------------
    # Get current path
    mypath = Path().absolute()
    mypath = os.path.abspath(os.path.join(mypath, os.pardir))
    train_path = mypath + "/dataTransformed/train_trans.csv"
    test_path = mypath + "/dataTransformed/test_trans.csv"
    price_path = mypath + "/dataTransformed/price_obs_dollars.csv"

    # Get dataset
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # price_df = file.inputs(price_path)

    print(train_df.head())
    print(test_df.head())

    # Plot price distribution
    distribution_Plots(train_df['SalePrice'], title='SalePrice', results_path='../resultsGraphs')
    distribution_Plots(train_df['SalePriceLog'], title='SalePriceLog', results_path='../resultsGraphs')

    # ------------ Prediction Model -----------------------------
    log = False
    print('log'.format(log))
    # Data split to train and validation data
    if log is True:
        price_obs = train_df.SalePriceLog.values
    else:
        price_obs = train_df.SalePrice.values
    train_df.drop(['SalePrice', 'SalePriceLog'], axis=1, inplace=True)

    train_score_l = []
    valid_score_l = []
    i = 0
    itr = 1 #100
    for i in range(itr):
        train_valid_X, train_valid_y = train_df, price_obs
        train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, train_size=.8)

        features_name = np.array(train_df.keys())

        model_params = {'n_estimators': 100}

        model = GradientBoostingRegressor(**model_params)
        model_name = 'Gradient Boosting Regressor Model'
        features = train_df.keys()
        price_pred, train_score, valid_score = modelTraining(model, model_name, features, train_X, train_y, valid_X,
                                                             valid_y, test_df, results_path)
        #trainingDeviance_Plot(model, model_params, model_name, valid_X, valid_y, results_path='../resultsGraphs')
        train_score_l.append(train_score)
        valid_score_l.append(valid_score)

    scores = {'trainScore': train_score_l, 'validScore': valid_score_l}
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(results_path + '/scores.csv')
    print(scores_df)
