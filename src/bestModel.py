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
from sklearn.model_selection import learning_curve
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
    price_pred_df.to_csv(results_path + '/' + model_name + '_price_pred_dollars.csv')

    # Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting
    print('\n****** ' + model_name + ' *********')
    print("Train score: {:.4f}".format(model.score(train_X, train_y)))
    print("Validation score: {:.4f}".format(model.score(valid_X, valid_y)))
    print("All training data score: {:.4f}".format(model.score(train_valid_X, train_valid_y)))

    return price_pred_df, model.score(train_X, train_y), model.score(valid_X, valid_y)


def bestModel(model, param_grid, n_jobs, train_valid_X, train_valid_y):
    model = GradientBoostingRegressor()
    #cv = ShuffleSplit(train_valid_X.shape[0], test_size=0.2)
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    classifier = GridSearchCV(estimator=model, cv=cv, param_grid=param_grid, n_jobs=n_jobs)

    classifier.fit(train_valid_X, train_valid_y)
    print("Best Estimator learned through GridSearch")
    print(classifier.best_estimator_)

    return cv, classifier.best_estimator_


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Nombre d'exemple d'entrainement")
    plt.ylabel("Taux de précision")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="g")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="g", label="Précision en entrainement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Précision validation")
    plt.legend(loc="best")
    return plt

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

    #print(train_df.head())
    #print(test_df.head())

    # Plot price distribution
    distribution_Plots(train_df['SalePrice'], title='SalePrice', results_path='../resultsGraphs')
    distribution_Plots(train_df['SalePriceLog'], title='SalePriceLog', results_path='../resultsGraphs')

    # ------------ Prediction Model -----------------------------
    log = False
    print('log = {}'.format(log))
    # Data split to train and validation data
    if log is True:
        price_obs = train_df.SalePriceLog.values
    else:
        price_obs = train_df.SalePrice.values
    train_df.drop(['SalePrice', 'SalePriceLog'], axis=1, inplace=True)

    train_valid_X, train_valid_y = train_df, price_obs
    #train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, train_size=.8)

    features_name = np.array(train_df.keys())

    param_grid = {'n_estimators': [100],
                  'learning_rate': [0.1],  # 0.05, 0.02, 0.01],
                  'max_depth': [6],  # 4,6],
                  'min_samples_leaf': [3],  # ,5,9,17],
                  'max_features': [1.0],  # ,0.3]#,0.1]
                  }
    '''
    param_grid = {'n_estimators': [100],
                  'learning_rate': [0.1, 0.05, 0.02, 0.01],
                  'max_depth': [2, 4, 6],
                  'min_samples_leaf': [3, 5, 9, 17],
                  'max_features': [1.0, 0.3, 0.1]
                  }
    '''
    param_grid = {'n_estimators': [100],
                  'learning_rate': [0.1],  # 0.05, 0.02, 0.01],
                  'max_depth': [6],  # 4,6],
                  'min_samples_leaf': [3],  # ,5,9,17],
                  'max_features': [0.3],  # ,0.3]#,0.1]
                  }

    n_jobs = 4
    model = GradientBoostingRegressor()
    cv, best_est = bestModel(model, param_grid, n_jobs, train_valid_X, train_valid_y)

    print("Best Estimator Parameters")
    print("n_estimators: %d" % best_est.n_estimators)
    print("max_depth: %d" % best_est.max_depth)
    print("Learning Rate: %.1f" % best_est.learning_rate)
    print("min_samples_leaf: %d" % best_est.min_samples_leaf)
    print("max_features: %.1f" % best_est.max_features)
    print("Train R-squared: %.2f" % best_est.score(train_valid_X, train_valid_y))

    estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth,
                                          learning_rate=best_est.learning_rate,
                                          min_samples_leaf=best_est.min_samples_leaf,
                                          max_features=best_est.max_features)

    title = ' Statistique '
    plot_learning_curve(estimator, title, train_valid_X, train_valid_y, cv=cv, n_jobs=n_jobs)
    plt.show()

    estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth,
                                          learning_rate=0.001,
                                          min_samples_leaf=best_est.min_samples_leaf,
                                          max_features=best_est.max_features)

    title = ' Statistique 0.001'
    plot_learning_curve(estimator, title, train_valid_X, train_valid_y, cv=cv, n_jobs=n_jobs)
    #plt.show()

    estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators,
                                          learning_rate=best_est.learning_rate)
    estimator.fit(train_valid_X, train_valid_y)
    impFeautures = SelectFromModel(estimator, prefit=True)
    X_new = impFeautures.transform(train_valid_X)

    title = ' Statistique '
    plot_learning_curve(estimator, title, X_new, train_valid_y, cv=cv, n_jobs=n_jobs)
    plt.show()
