#Vikram Gokhale
#predictor for Premier League soccer matches

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/vikra/PLMP/PremierLeague_Cleaned_Final_Fixed.csv')

df = df.iloc[10:]

df.drop('AwayTeamPoints', axis = 1, inplace = True)

#mapping 3 points to a win and 1 and 0 points to dropped points
df['HomeTeamPoints'] = df['HomeTeamPoints'].map({3: 'W', 1: 'DP', 0: 'DP'})

df.head()

#creating a column containing which team has better B365 odds to feed the model
df['expected_winner'] = df.apply(lambda row: 1 if (row['B365AwayTeam'] - row['B365HomeTeam']) > 0 else 0, axis=1)

df.tail()

#creating a dictionary of all the matches
matchdays_dict = {}
count = 1
for i in df['Date']:
    if i not in matchdays_dict:
        matchdays_dict[i] = count
        count += 1

matchdays_dict['2025-02-27'] 

#storing number of matchdays
matchdays = 0
for i in matchdays_dict.keys():
    matchdays += 1

matchdays

df['recency'] = 0 #initializing column for recency

df.head()

#mapping a value between 0 and 1 for how recent each game is
for i in range(len(df)):
    if df.iloc[i, 3] in matchdays_dict:
        df.iloc[i, 31] = matchdays_dict[df.iloc[i, 3]] / matchdays

df.iloc[1200, 31] 

#breaking of recency column to use as weights for each data point when training the model
weights = df['recency']

df = df.drop('recency', axis=1)

df.tail()

weights

#dropping non-numerical columns
df.drop(['MatchID', 'Season', 'MatchWeek', 'Date', 'HomeTeam', 'AwayTeam'], axis=1, inplace = True)

df.tail()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#defining range of hyperparameters to tune by grid search
n_estimators=[50, 100, 200, 300, 500]
bootstrap = [True,False]
max_depth = [None, 5, 10, 20]
max_features = ['sqrt', 'log2', None]

param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'bootstrap': bootstrap, 'max_depth': max_depth}

rfc = RandomForestClassifier(class_weight='balanced')
grid = GridSearchCV(rfc, param_grid, cv=6, n_jobs = -1, verbose = 1)

#splitting data into train and test sections ensuring test data is more recent
df = df.dropna()
test_size = 0.15
split_index = int((1 - test_size) * len(df))

X_train = df.drop('HomeTeamPoints', axis=1)[:split_index]
X_test = df.drop('HomeTeamPoints', axis=1)[split_index:]

y_train = df['HomeTeamPoints'][:split_index]
y_test = df['HomeTeamPoints'][split_index:]

y_train #confirming correctly ordered data

weights2 = weights[:split_index]

weights2

#fitting training data and their weights to model
grid.fit(X_train, y_train, sample_weight = weights2)

grid.best_params_

preds = grid.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#testing accuracy
confusion_matrix(y_test, preds)

print(accuracy_score(y_test, preds))

print(classification_report(y_test,preds))

preds

print(preds != y_test)

y_test

comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': preds
})

analysis = comparison_df.tail(50)

analysis