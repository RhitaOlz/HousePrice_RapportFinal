import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Modelling Algorithms
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgbm
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.metrics import mean_squared_error

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

def GradientBoostingRegressor_Plots(X_train, y_train, X_test, y_test, var_index):

    # #############################################################################
    # Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    gbr_model = GradientBoostingRegressor(**params)

    gbr_model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, gbr_model.predict(X_test))
    print("MSE: %.4f" % mse)

    # #############################################################################
    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(gbr_model.staged_predict(X_test)):
        test_score[i] = gbr_model.loss_(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, gbr_model.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # #############################################################################
    # Plot feature importance
    feature_importance = gbr_model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, var_index[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    #plt.show()

    # Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting
    print('\n****** GradientBoostingRegressor Model Specific *********')
    print("Train score: {:.4f}".format(model.score(X_train, y_train)))
    print("Validation score: {:.4f}".format(model.score(X_test, y_test)))
    #print("All training data score: {:.4f}".format(model.score(train_valid_X, train_valid_y)))

def rmsle_cv(model, train_df, train_y):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)
    rmse = np.sqrt(-cross_val_score(model, train_df.values, train_y, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

def feauturesImportance_Plot(model, model_name, features_name, results_path='../resultsGraphs'):
    fig = plt.figure()

    # Plot feature importance
    feature_importance = model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    size = 40
    sorted_idx = sorted_idx[len(features_name)-size:]
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, features_name[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    #plt.show()

    fig.savefig(results_path + '/' + model_name + '_' + str(size) +'importantFeatures')

def trainingDeviance_Plot(model, params, model_name, valid_X, valid_y, results_path='../resultsGraphs'):
    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    if model_name in ['Random Forest Regressor Model', 'XGboostingModel']:
        y_preds = model.predict(valid_X)
        for i, y_pred in enumerate(y_preds):
            test_score[i] = model.loss(valid_y, y_pred)
    else:
        y_preds = model.staged_predict(valid_X)
        for i, y_pred in enumerate(y_preds):
            test_score[i] = model.loss_(valid_y, y_pred)

    fig = plt.figure()

    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, model.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    fig.savefig(results_path + '/' + model_name + '_Deviance')

def modelTraining(model, model_name, features_name, train_X, train_y, valid_X, valid_y, test_df, results_path):
    # model train
    start = time.time()
    model.fit(train_X, train_y)
    end = time.time()

    # Feature selection from model
    impFeature = SelectFromModel(model, prefit=True)
    X_new = impFeature.transform(train_X)
    print('Feature selection from model')
    print('new features shape {}'.format(X_new.shape))
    feauturesImportance_Plot(model, model_name, features_name, results_path)

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

    # Validation function
    if log is True:
        score = rmsle_cv(model, train_df, price_obs)
        print("\n{} score: {:.4f} ({:.4f})".format(model_name, score.mean(), score.std()))
    print('{} time: {:.2f}s\n'.format(model_name, end - start))

    # Resutls plots
    predictedPrice_Plots(model_name, price_pred, price_obs[:len(price_pred)])
    predictedPrice_Plots(model_name, model.predict(valid_X), valid_y)

    #crossValidation(model, valid_X, valid_y)

    return price_pred_df, model.score(train_X, train_y), model.score(valid_X, valid_y)

def crossValidation(model, valid_X, valid_y):
    # Cross validation
    scoring = ['precision_macro', 'recall_macro']
    CV_scores = cross_validate(model, valid_X, valid_y, scoring=scoring, cv=5, return_train_score=False)

    print('crose validation score {}'.format(CV_scores))

def modelOptimumTraining(model, model_name, features_name, train_X, train_y, valid_X, valid_y, test_df, results_path):
    # model train
    start = time.time()
    model.fit(train_X, train_y)
    end = time.time()

    # Feature selection from model
    impFeature = SelectFromModel(model, prefit=True)
    X_new = impFeature.transform(train_X)
    print('Feature selection from model')
    print('new features shape {}'.format(X_new.shape))

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
    #print("All training data score: {:.4f}".format(model.score(train_valid_X, train_valid_y)))

    print('{} time: {:.2f}s\n'.format(model_name, end - start))

    return price_pred_df


if __name__ == '__main__':

    # ------------ Results folders ---------------------------------
    results_path = '../resultsGraphs'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    make_dir(results_path)

    results_optim_path = '../optim_resultsGraphs'
    if not os.path.exists(results_optim_path):
        os.makedirs(results_optim_path)
    make_dir(results_optim_path)

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
    log = True
    print('log = {}'.format(log))
    # Data split to train and validation data
    if log is True:
        price_obs = train_df.SalePriceLog.values
    else:
        price_obs = train_df.SalePrice.values
    train_df.drop(['SalePrice', 'SalePriceLog'], axis=1, inplace=True)

    train_valid_X, train_valid_y = train_df, price_obs
    train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, train_size=.8)

    features_name = np.array(train_df.keys())

    '''    
    ##### XGboosting Model ####

    model_params = {'n_estimators': 100}
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    model_name = 'XGboostingModel'
    features = train_df.keys()
    price_pred, train_score, valid_score = modelOptimumTraining(model, model_name, features, train_X, train_y, valid_X,
                                                         valid_y, test_df, results_path)
    trainingDeviance_Plot(model, model_params, model_name, valid_X, valid_y, results_path='../resultsGraphs')

    # Optimisation
    model_name = 'XGboosting Model White Importent Fetures'
    impFeautures = SelectFromModel(model, prefit=True)
    train_X_new = impFeautures.transform(train_X)
    valid_X_new = impFeautures.transform(valid_X)
    test_X_new = impFeautures.transform(test_df)

    print('New train_X shape {}'.formatstaged_predict(train_X_new.shape))
    price_pred = modelOptimumTraining(model, model_name, features_name, train_X_new, train_y, valid_X_new, valid_y,
                                      test_X_new,
                                      results_optim_path)
    
    '''

    ##### Random Forest Regressor Model ####
    model_params = {'n_estimators': 100}
    model = RandomForestRegressor(**model_params)
    model_name = 'Random Forest Regressor Model'
    price_pred, train_score, valid_score = modelTraining(model, model_name, features_name, train_X, train_y, valid_X,
                                                         valid_y, test_df.values, results_path)

    # trainingDeviance_Plot(model, model_params, model_name, valid_X, valid_y, results_path='../resultsGraphs')

    # Optimisation
    model_name = 'Random Forest Regressor Model White Importent Fetures'
    impFeautures = SelectFromModel(model, prefit=True)
    train_X_new = impFeautures.transform(train_X)
    valid_X_new = impFeautures.transform(valid_X)
    test_X_new = impFeautures.transform(test_df)

    print('New train_X shape {}'.format(train_X_new.shape))
    price_pred = modelOptimumTraining(model, model_name, features_name, train_X_new, train_y, valid_X_new, valid_y,
                                      test_X_new,
                                      results_optim_path)

    ##### Gradient Boosting Regressor Model ####

    model_params = {'n_estimators': 100}
    model = GradientBoostingRegressor(**model_params)
    model_name = 'Gradient Boosting Regressor Model'
    features = train_df.keys()
    price_pred, train_score, valid_score = modelTraining(model, model_name, features, train_X, train_y, valid_X,
                                                         valid_y, test_df, results_path)
    trainingDeviance_Plot(model, model_params, model_name, valid_X, valid_y, results_path='../resultsGraphs')

    # Optimisation
    model_name = 'Gradient Boosting Regressor Model White Importent Fetures'
    impFeautures = SelectFromModel(model, prefit=True)
    train_X_new = impFeautures.transform(train_X)
    valid_X_new = impFeautures.transform(valid_X)
    test_X_new = impFeautures.transform(test_df)

    print('New train_X shape {}'.format(train_X_new.shape))
    price_pred = modelOptimumTraining(model, model_name, features_name, train_X_new, train_y, valid_X_new, valid_y,
                                      test_X_new,
                                      results_optim_path)
    # GradientBoostingRegressor_Plots(train_X, train_y, valid_X, valid_y, train_df.keys())

    ##### Light Gradient Boosting Machine Model ####

    model_params = {'n_estimators': 100}
    model = lgbm.LGBMRegressor(**model_params)
    model_name = 'Light Gradient Boosting Regressor Model'
    features = train_df.keys()
    price_pred, train_score, valid_score = modelTraining(model, model_name, features, train_X, train_y, valid_X,
                                                         valid_y, test_df, results_path)

    # trainingDeviance_Plot(model, model_params, model_name, valid_X, valid_y, results_path='../resultsGraphs')
    # distribution_Plots(price_pred, title='PredictedSalePrice', results_path='../resultsGraphs')

    # Optimisation
    model_name = 'Light Gradient Boosting Regressor Model White Importent Fetures'
    impFeautures = SelectFromModel(model, prefit=True)
    train_X_new = impFeautures.transform(train_X)
    valid_X_new = impFeautures.transform(valid_X)
    test_X_new = impFeautures.transform(test_df)

    print('New train_X shape {}'.format(train_X_new.shape))
    price_pred = modelOptimumTraining(model, model_name, features_name, train_X_new, train_y, valid_X_new, valid_y,
                                      test_X_new,
                                      results_optim_path)

    ##### Light Gradient Boosting Machine Model Optimisation ####
    '''
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

    #distribution_Plots(price_pred, title='PredictedSalePrice', results_path='../resultsGraphs')
    '''


    #plt.show()




