import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Algorithms
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
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


def plot_samples(S, axis_list=None):
    plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
                color='steelblue', alpha=0.5)
    if axis_list is not None:
        colors = ['orange', 'red']
        for color, axis in zip(colors, axis_list):
            axis /= axis.std()
            x_axis, y_axis = axis
            # Trick to get legend to work
            plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
            plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,
                       color=color)

    plt.hlines(0, -3, 3)
    plt.vlines(0, -3, 3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x')
    plt.ylabel('y')

def IndependentFeatre_Slection(X, y, ):
    bestFeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestFeatures.fit(X, y)
    fscores = pd.DataFrame(fit.scores_)
    fcolumns = pd.DataFrame(X.columns)
    return bestFeatures

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
    price_df = pd.read_csv(price_path)

    print(train_df.head())
    print(test_df.head())

    # Plot price distribution
    distribution_Plots(train_df['SalePrice'], title='SalePrice', results_path='../resultsGraphs')
    distribution_Plots(train_df['SalePriceLog'], title='SalePriceLog', results_path='../resultsGraphs')

    # ------------ Variables importantes --------------------------------------

    y = price_df.values
    train_df.drop(['SalePrice', 'SalePriceLog'], axis=1, inplace=True)
    X = train_df.values

    n_components = [5] #[5, 10, 20]
    svd_solver = ['auto'] # ['auto', 'full', 'arpack', 'randomized']

    for i in n_components:
        for solver in svd_solver:
            '''
            pca = PCA(n_components=n_components, svd_solver=solver)
            pca.fit_transform(train_df.values)
            var = pca.explained_variance_ratio_
            singular = pca.singular_values_
            components = pca.components_
            '''

            #rng = np.random.RandomState(42)
            #S = rng.standard_t(1.5, size=(20000, 2))
            X = shuffle(X, random_state=0)

            pca = PCA()
            S_pca_ = pca.fit(X).transform(X)


            '''
            ica = FastICA(random_state=0)
            S_ica_ = ica.fit(X).transform(X)  # Estimate the sources
            
            S_ica_ /= S_ica_.std(axis=0)
            plt.figure()
            plt.subplot(2, 2, 1)
            plot_samples(X / X.std())
            plt.title('True Independent Sources')
            
            axis_list = [pca.components_.T, ica.mixing_]
            plt.subplot(2, 2, 2)
            plot_samples(X / np.std(X), axis_list=axis_list)
            legend = plt.legend(['PCA', 'ICA'], loc='upper right')
            legend.set_zorder(100)

            plt.title('Observations')
            '''
            plt.subplot(2, 2, 3)
            plot_samples(S_pca_ / np.std(S_pca_, axis=0))
            plt.title('PCA recovered signals')
            '''
            plt.subplot(2, 2, 4)
            plot_samples(S_ica_ / np.std(S_ica_))
            plt.title('ICA recovered signals')
            '''
            plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
            plt.show()



