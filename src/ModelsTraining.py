import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DataStat import Handeler
from pathlib import Path

# Modelling Algorithms
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgbm

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew

from dataTransform import distribution_Plots

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def vadiables_Plots(varibles, missingData_size, title='Variables', results_path='../resultsGraphs'):
    fig = plt.figure()

    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=varibles, y=missingData_size)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)

    fig.savefig(results_path + '/' + title + '_distrubution')

def predictedPrice_Plots(model_name, price_pred, price_obs, results_path='../resultsGraphs'):
    fig = plt.figure()

    plt.scatter(price_pred, price_obs, alpha=1, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Observed Price')
    plt.title(model_name)
    overlay = '' #infoModel
    plt.annotate(s=overlay, xy=(12.1, 10.6), size='x-large')

    fig.savefig(results_path + '/' + model_name + '_distrubution')

def rmsle_cv(model, train_df, train_y):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)
    rmse = np.sqrt(-cross_val_score(model, train_df.values, train_y, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

if __name__ == '__main__':

    # ------------ Results folder ---------------------------------
    results_path = '../resultsGraphs'
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
    #price_df = file.inputs(price_path)

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

    train_valid_X, train_valid_y = train_df, price_obs
    train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, train_size=.8)

    ##### Random Forest Regressor Model ####
    model = RandomForestRegressor()

    start = time.time()
    model.fit(train_X, train_y)
    end = time.time()

    # Prediction
    price_pred = model.predict(test_df)
    # Price value save
    if log is True:
        np.savetxt("RandomForestRegressor_price_pred_log.csv", price_pred, delimiter=",")
        price_pred = [np.round(i, 9) for i in np.expm1(price_pred)]

    price_pred_df = pd.DataFrame(price_pred)
    price_pred_df.to_csv(results_path + '/RandomForestRegressor_price_pred_dollars.csv')
    #np.savetxt("RandomForestRegressor_price_pred_dollars.csv", price_pred, delimiter=",")

    # Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting
    print('\n****** RandomForestRegressor *********')
    print("Train score: {:.4f}".format(model.score(train_X, train_y)))
    print("Validation score: {:.4f}".format(model.score(valid_X, valid_y)))
    print("All training data score: {:.4f}".format(model.score(train_valid_X, train_valid_y)))

    # Validation function
    score = rmsle_cv(model, train_df, price_obs)
    print("\nRandomForestRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    print('RandomForestRegressor Training time: {:.2f}s\n'.format(end - start))

    # Resutls plots
    predictedPrice_Plots('Random Forest Regressor Model', price_pred, price_obs[:len(price_pred)])
    predictedPrice_Plots('Random Forest Regressor Model', model.predict(valid_X), valid_y)


    ##### Gradient Boosting Regressor Model ####
    model = GradientBoostingRegressor()

    start = time.time()
    model.fit(train_X, train_y)
    end = time.time()

    # Prediction
    price_pred = model.predict(test_df)
    # Price value save
    if log is True:
        np.savetxt("GradientBoostingRegressor_price_pred_log.csv", price_pred, delimiter=",")
        price_pred = [np.round(i, 9) for i in np.expm1(price_pred)]
    price_pred_df = pd.DataFrame(price_pred)
    price_pred_df.to_csv(results_path + '/GradientBoostingRegressor_price_pred_dollars.csv')
    #np.savetxt("GradientBoostingRegressor_price_pred_dollars.csv", price_pred, delimiter=",")

    # Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting
    print('\n****** GradientBoostingRegressor *********')
    print("Train score: {:.4f}".format(model.score(train_X, train_y)))
    print("Validation score: {:.4f}".format(model.score(valid_X, valid_y)))
    print("All training data score: {:.4f}".format(model.score(train_valid_X, train_valid_y)))

    # Validation function
    score = rmsle_cv(model, train_df, price_obs)
    print("\nGradientBoostingRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    print('GradientBoostingRegressor Training time: {:.2f}s\n'.format(end - start))

    predictedPrice_Plots('Gradient Boosting Regressor Model', price_pred, price_obs[:len(price_pred)])
    predictedPrice_Plots('Gradient Boosting Regressor Model', model.predict(valid_X), valid_y)

    ##### Light Gradient Boosting Machine Model ####

    model = lgbm.LGBMRegressor()

    start = time.time()
    model.fit(train_X, train_y)
    end = time.time()

    # Prediction
    price_pred = model.predict(test_df)
    # Price value save
    if log is True:
        np.savetxt("LGBMRegressorInitial_price_pred_log.csv", price_pred, delimiter=",")
        price_pred = [np.round(i, 9) for i in np.expm1(price_pred)]

    price_pred_df = pd.DataFrame(price_pred)
    price_pred_df.to_csv(results_path + '/LGBMRegressorInitial_price_pred_dollars.csv')
    #np.savetxt("LGBMRegressorInitial_price_pred_dollars.csv", price_pred, delimiter=",")

    # Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting
    print('\n****** LGBMRegressorInitial *********')
    print("Train score: {:.4f}".format(model.score(train_X, train_y)))
    print("Validation score: {:.4f}".format(model.score(valid_X, valid_y)))
    print("All training data score: {:.4f}".format(model.score(train_valid_X, train_valid_y)))

    # Validation function
    score = rmsle_cv(model, train_df, price_obs)
    print("\nLGBMRegressorInitial score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    print('LGBMRegressorInitial Training time: {:.2f}s\n'.format(end - start))

    predictedPrice_Plots('Light Gradient Boosting Machine Model', price_pred, price_obs[:len(price_pred)])
    predictedPrice_Plots('Light Gradient Boosting Machine Model', model.predict(valid_X), valid_y)

    distribution_Plots(price_pred, title='PredictedSalePrice', results_path='../resultsGraphs')

    ##### Light Gradient Boosting Machine Model Optimisation ####

    print('\n****** LGBMRegressor Optimisation *********')
    model = lgbm.LGBMRegressor()

    start = time.time()
    model.fit(train_X, train_y,
            eval_set=[(valid_X, valid_y)],
            eval_metric='l1',
            early_stopping_rounds=5)
    end = time.time()

    price_pred = model.predict(test_df)
    # Price value save
    if log is True:
        np.savetxt("LGBMRegressor_price_pred_log.csv", price_pred, delimiter=",")
        price_pred = [np.round(i, 9) for i in np.expm1(price_pred)]
    price_pred_df = pd.DataFrame(price_pred)
    price_pred_df.to_csv(results_path + '/LGBMRegressor_price_pred_dollars.csv')

    #np.savetxt(".csv", price_pred, delimiter=",")

    # Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting
    print("Train score: {:.4f}".format(model.score(train_X, train_y)))
    print("Validation score: {:.4f}".format(model.score(valid_X, valid_y)))
    print("All training data score: {:.4f}".format(model.score(train_valid_X, train_valid_y)))

    # Validation function
    score = rmsle_cv(model, train_df, price_obs)
    print("\nLGBMRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    print('LGBMRegressor Training time: {:.2f}s\n'.format(end - start))

    predictedPrice_Plots('Light Gradient Boosting Machine Model', price_pred, price_obs[:len(price_pred)])
    predictedPrice_Plots('Light Gradient Boosting Machine Model', model.predict(valid_X), valid_y)

    distribution_Plots(price_pred, title='PredictedSalePrice', results_path='../resultsGraphs')




