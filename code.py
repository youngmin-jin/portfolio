# based on Google colab

# --------------------------------------------------------------
# import modules
# --------------------------------------------------------------
# analysis
import numpy as np
import pandas as pd

# file upload
from google.colab import files
import io

# visualization
import matplotlib.pyplot as plt
import missingno 
from missingno.missingno import heatmap
from pandas.plotting import boxplot

# --------------------------------------------------------------
# read and check data
# --------------------------------------------------------------
# read data
uploaded = files.upload()
df_analysis = pd.read_csv(io.BytesIO(uploaded['acoustic_analysis.csv']), encoding='utf-8')
df_details = pd.read_csv(io.BytesIO(uploaded['details.csv']), encoding='utf-8')

# check data 
# df_analysis
df_analysis.head(10)
df_analysis.describe()
df_analysis.dtypes
df_analysis.nunique()

boxplot(df_analysis['danceability'])
boxplot(df_analysis['acousticness'])
boxplot(df_analysis['speechiness'])
df_analysis.isnull().sum()
missingno.matrix(df_analysis)

# df_details
df_details.head(10)
df_details.describe()
df_details.dtypes
df_details.isnull().sum()

# --------------------------------------------------------------
# restructure and cleanse the data
# --------------------------------------------------------------
# merge two datasets
df = pd.concat([df_details, df_analysis.drop('id', axis=1)], axis=1)

# delete the duplicated column and change the order
df_cols = df.columns
df_cols = ['id','title', 'artist', 'genre', 'year', 'bpm', 'dB', 'energy', 'danceability', 'liveness', 'valence', 'duration',
       'acousticness', 'speechiness', 'popularity']
df = df[df_cols]

# dummy categories
# split title values
df['title'] = df['title'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
title_split = df['title'].str.split(pat=' ', expand=True)
df.drop('title', axis=1, inplace=True)
df = pd.concat([df, title_split], axis=1)

# encode string values before splitting into training and test dataset
col = title_split.columns
categorial_cols = df[col]
categorial_cols = pd.concat([categorial_cols, df[['genre','artist']]], axis=1)

df_dummies = pd.get_dummies(categorial_cols, drop_first=True)
df.drop(list(categorial_cols), axis=1, inplace=True)
df = pd.concat([df, df_dummies], axis=1)

# handle invalid data
# remove outliers
df = df[df['dB']!=-60]
df = df[df['popularity']!=0]

# check correlation
df_temp = df.drop(['id'], axis=1)
df_corr = df_temp.corr()

# extract correlation that has the most high num with popularity
df_corr_popularity = pd.DataFrame(df_corr['popularity'].drop('popularity', axis=0))
df_corr_popularity = df_corr_popularity.sort_values('popularity', ascending=False)
df_corr_popularity[df_corr_popularity['popularity']>0.1]
high_corr_cols = list(df_corr_popularity[df_corr_popularity['popularity']>0.1].index)
high_corr_cols

# visualize correlation 
def visualize_with_popularity(df, col):
  temp = pd.DataFrame(df.groupby(col)['popularity'].agg('median'))
  plt.plot(temp.index, temp['popularity'])
  plt.title(col)
  plt.show()

for col in high_corr_cols:
  visualize_with_popularity(df, col)

# fill null values
# save columns that have null values
df_null_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

# fill null values with median of each year
for col in df_null_cols:
  df[col] = df_original.groupby(['year'])[col].transform(lambda x: x.fillna(x.median())) 

# --------------------------------------------------------------
# split into training and test datasets
# --------------------------------------------------------------
from sklearn.model_selection import train_test_split
df_independent = df.drop(['popularity'], axis=1)
df_dependent = df[['popularity']]

# use shuffle as it is ordered by year
x_train, x_test, y_train, y_test = train_test_split(df_independent, df_dependent, test_size=0.2, shuffle=True)

# --------------------------------------------------------------
# modelling
# --------------------------------------------------------------
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# return r2 score, rmse, predicted and actual values
def return_results(model):
  y_predicted = model.predict(x_test)
  y_actual = np.array(y_test['popularity'])

  r2_score_results = r2_score(y_actual, y_predicted)
  print('r2: ', r2_score_results)

  rmse_results = mean_squared_error(y_actual, y_predicted, squared=False)
  print('rmse: ', rmse_results)

  y_predicted = pd.DataFrame(y_predicted, columns=['predicted'])
  y_actual = pd.DataFrame(y_actual, columns=['actual'])
  y_results = pd.concat([y_predicted, y_actual], axis=1)

  return y_results

# --------------- linear regression ---------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# initiate linear regression 
reg1 = LinearRegression()

# cross-validation
reg1_folds = KFold(n_splits = 10, shuffle = True, random_state = 100)
reg1_scores = cross_val_score(reg1, x_train, y_train, scoring='r2', cv=reg1_folds)
reg1_scores

# fit
reg1_model = reg1.fit(x_train, y_train)

# evaluate
return_results(reg1_model)

# --------------- random forest regression ---------------
# with grid search and hyperparameter tuning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# initiate the rf
rf2 = RandomForestRegressor()

# cross-validation
rf2_folds = KFold(
    n_splits=10
    , shuffle=True
    , random_state=100
)

# hyperparams
hyper_params = {
    'n_estimators':[25,50,100,150]
    , 'max_features':['sqrt','log2',None]
    , 'max_depth':[5,10,15]
    , 'max_leaf_nodes':[5,10,15]
}

# grid search 
rf2_model = GridSearchCV(
    estimator = rf2
    , param_grid = hyper_params
    , scoring = 'r2'
    , cv = rf2_folds
    , verbose = 1
    , return_train_score =True
)

# fit
rf2_model.fit(x_train, y_train)

# evaluate
return_results(rf2_model)

# --------------- XGBoost regression ---------------
# with grid search and hyperparameter tuning
# initate the model
xg2 = xg.XGBRegressor()

# hyperparams
params={ 
    'n_estimators':[800,900,950,1000]             
    , 'max_depth': [3,5,7,10]
    , 'colsample_bylevel': [0.7,0.9,0.95]
    , 'learning_rate': [0.01, 0.03, 0.05, 0.07]
}

# grid search cv
from sklearn.model_selection import GridSearchCV
xg2_model = GridSearchCV(
    estimator=xg2
    , param_grid=params
    , cv=2
    , n_jobs=-1
)

# fit
xg2_model.fit(x_train, y_train)

# evaluate
return_results(xg2_model)

# check best parameters
xg2_model.best_estimator_

# get predicted values
y_predicted = xg2_model.predict(x_test)
df_xg2 = pd.concat([x_test, y_test], axis=1)
df_xg2['predicted'] = y_predicted

# visualize the actual values and predicted values
df_xg2 = return_results(xg2_model)
plt.scatter(df_xg2['actual'], df_xg2['predicted'])
plt.xlabel('Actual popularity')
plt.ylabel('Predicted popularity')
plt.plot(np.unique(df_xg2['actual']), np.poly1d(np.polyfit(df_xg2['actual'], df_xg2['predicted'], 1))(np.unique(df_xg2['actual'])))
plt.show()
