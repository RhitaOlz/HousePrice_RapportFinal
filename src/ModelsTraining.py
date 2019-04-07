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

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def distribution_Plots(distribution, title='SalePrice', results_path='../resultsGraphs'):# Distribution parameters
    (mu, sigma) = norm.fit(distribution)
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Plot the distribution
    fig = plt.figure()
    sns.distplot(distribution, fit=norm)
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title(title + ' distribution')
    fig.savefig(results_path + '/' + title + '_distrubution')

    # Pot the QQ-plot
    fig = plt.figure()
    res = stats.probplot(distribution, plot=plt)
    fig.savefig(results_path + '/' + title + '_QQ-plot')
    #plt.show()

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
    return (rmse)
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
    train_path = mypath + "/data/train.csv"
    test_path = mypath + "/data/test.csv"

    # Get dataset
    file = Handeler()
    train_df = file.inputs(train_path)
    test_df = file.inputs(test_path)

    print(train_df.head())
    print(test_df.head())

    # Plot price distribution
    distribution_Plots(train_df['SalePrice'], title='SalePrice', results_path='../resultsGraphs')

    # ------------ Price normalisation -----------------------------
    # Log transformation of sale price
    train_df["SalePriceLog"] = np.log1p(train_df["SalePrice"])
    print(train_df.head())

    # Plot SalePriceLog distribution
    distribution_Plots(train_df['SalePriceLog'], title='SalePriceLog', results_path='../resultsGraphs')

    # ------------ Variables analysis -----------------------------
    # all data : train and test
    train_size = train_df.shape[0]
    test_size = test_df.shape[0]
    price_obs = train_df.SalePriceLog.values
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    all_data.drop(['SalePrice', 'SalePriceLog'], axis=1, inplace=True)
    print(all_data.head())
    print("all_data size is : {}".format(all_data.shape))

    # Missing data
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    missing_data.head(20)
    vadiables_Plots(all_data_na.index, all_data_na)

    # Correlation map to see how features are correlated with SalePrice
    corrmat = train_df.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)

    # ------------ Data augmentation -----------------------------
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    # Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

    # Check remaining missing values if any
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    print(missing_data.head())

    # ------------ Variables transformation -----------------------------
    # MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    # Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    # Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # Label Encoding some categorical variables that may contain information in their ordering set
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))

    # shape
    print('Shape all_data: {}'.format(all_data.shape))

    # Adding total sqfootage feature
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    '''
    # Check the skew of all numerical features ??????????????????666
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    print(skewness.head(10))
    '''

    # Getting dummy categorical features
    all_data = pd.get_dummies(all_data)
    print(all_data.shape)

    # Get transformed data
    train_df = all_data[:train_size]
    test_df = all_data[train_size:]

    train_df.to_csv(results_path + '/train_df.csv')
    test_df.to_csv(results_path + '/test_df.csv')
    pd.DataFrame(price_obs).to_csv(results_path + '/price_obs_log.csv')
    # Price value in $
    price_obs_dollars = np.expm1(price_obs)
    pd.DataFrame(price_obs_dollars).to_csv(results_path + '/price_obs_dollars.csv')

    np.savetxt("price_obs.csv", price_obs, delimiter=",")

    # ------------ Prediction Model -----------------------------

    # Data split to train and validation data
    keys = train_df.keys()
    train_valid_X, train_valid_y = train_df[keys], price_obs
    train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, train_size=.8)

    ##### Random Forest Regressor Model ####
    model = RandomForestRegressor()

    start = time.time()
    model.fit(train_X, train_y)
    end = time.time()

    # Get prediction values
    price_pred = model.predict(test_df)
    np.savetxt("RandomForestRegressor_price_pred_log.csv", price_pred, delimiter=",")
    # Price value in $
    price_pred_dollars = np.expm1(price_pred)
    np.savetxt("RandomForestRegressor_price_pred_dollars.csv", price_pred_dollars, delimiter=",")

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

    price_pred = model.predict(test_df)
    np.savetxt("GradientBoostingRegressor_price_pred.csv", price_pred, delimiter=",")
    # Price value in $
    price_pred_dollars = np.expm1(price_pred)
    np.savetxt("GradientBoostingRegressor_price_pred_dollars.csv", price_pred_dollars, delimiter=",")

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

    price_pred = model.predict(test_df)
    np.savetxt("LGBMRegressor_price_pred.csv", price_pred, delimiter=",")
    # Price value in $
    price_pred_dollars = np.expm1(price_pred)
    np.savetxt("LGBMRegressor_price_pred_dollars.csv", price_pred_dollars, delimiter=",")

    # Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting
    print('\n****** LGBMRegressor *********')
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

    plt.show()



