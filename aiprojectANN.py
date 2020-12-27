# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:16:21 2020

@author: reine
"""
#import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error
import seaborn as sns

#import dataset
qb_dataset = pd.read_csv('D:/Documents/NFL Data/qb_data2.csv')

#create new dataframe with features
df = qb_dataset[['player_team_score',
                 'opponent_score',
                 'avg_player_passing_attempts',
                 'avg_player_passing_completions',
                 'avg_player_passing_yards',
                 'avg_player_passing_touchdowns',
                 'avg_player_interceptions',
                 'avg_player_passing_2pt_conversions',
                 'avg_player_rushing_attempts',
                 'avg_player_rushing_yards',
                 'avg_player_rushing_touchdowns',
                 'avg_player_fumbles',
                 'def_pass_attempts_allowed',
                 'def_pass_yards_allowed',
                 'def_pass_tds_allowed',
                 'def_interceptions',
                 'def_2point_pass_conversion_allowed',
                 'def_rushing_attempts_allowed',
                 'def_rushing_yards_allowed',
                 'def_rushing_tds_allowed',
                 'def_2point_rush_conversion_allowed',
                 'def_draftkings_fpp_allowed',
                 'player_draftkings_fantasypoints'
                 ]]

print(df.head())
print(df.isnull().any())

#split data into train/test sets
X = df.iloc[:, 0:22].values
y = df.iloc[:, 22].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(X_test)

# <<<<< MAKE THE ANN >>>>>
#init the ANN
model = Sequential()

#add input layer & first hidden layer
model.add(Dense(units=6, kernel_initializer='normal', activation='relu', input_dim=22))
model.add(Dropout(rate=0.01))

i = 0
while i < 2:
    model.add(Dense(units=6, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(rate=0.01))
    i += 1

#add output layer
model.add(Dense(units=1, kernel_initializer='normal', activation='linear'))

#complie ANN
model.compile(optimizer='rmsprop', loss='mean_absolute_error', metrics=['mean_absolute_error','accuracy'])

#fit ANN to Training
model.fit(X_train, y_train, batch_size = 50, epochs = 1000)
#model.fit(X_train, y_train, batch_size = 100, epochs = 100)

#make predictions
y_pred = model.predict(X_test)

y_predicted = []
for y in y_pred:
    y_predicted.append(y[0])
expected = y_test

#Plot visualization of expected vs predicted
df3 = pd.DataFrame()
df3['Expected'] = pd.Series(expected)
df3['Predicted'] = pd.Series(y_predicted)
figure = plt.figure(figsize=(11,11))
axes = sns.scatterplot(data=df3, x='Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)
start=min(expected.min(), y_pred.min())
end=max(expected.max(), y_pred.max())
axes.set_xlim(start,end+1)
axes.set_ylim(start,end+1)
line=plt.plot([start,end],[start,end],'k--')

print(f'Mean Absolute Error of Test Set: {mean_absolute_error(y_test, y_pred)}')

y = 0
count = 1
total = 0
lessthan2 = 0
lessthan5 = 0
for z in expected:
    if y_predicted[y] >= z:
        total += 1
    if abs(z-y_predicted[y]) <= 2:
        lessthan2 += 1
    if abs(z-y_predicted[y]) <= 5:
        lessthan5 += 1
    count += 1
    y +=1

percent = total/count
percent2 = lessthan2 / count
percent5 = lessthan5 / count
print(f'The percent of predictions >= expected value: {percent}')
print(f'The percent of predictions <= 2 points of expected value: {percent2}')
print(f'The percent of predictions <= 5 points of expected value: {percent5}')
    

superbowl_df = pd.read_csv('D:/Documents/NFL Data/patmahomes.csv')
X_superbowl = superbowl_df[['player_team_score',
                            'opponent_score',
                            'avg_player_passing_attempts',
                            'avg_player_passing_completions',
                            'avg_player_passing_yards',
                            'avg_player_passing_touchdowns',
                            'avg_player_interceptions',
                            'avg_player_passing_2pt_conversions',
                            'avg_player_rushing_attempts',
                            'avg_player_rushing_yards',
                            'avg_player_rushing_touchdowns',
                            'avg_player_fumbles',
                            'def_pass_attempts_allowed',
                            'def_pass_yards_allowed',
                            'def_pass_tds_allowed',
                            'def_interceptions',
                            'def_2point_pass_conversion_allowed',
                            'def_rushing_attempts_allowed',
                            'def_rushing_yards_allowed',
                            'def_rushing_tds_allowed',
                            'def_2point_rush_conversion_allowed',
                            'def_draftkings_fpp_allowed']]

y_superbowl = superbowl_df[['player_draftkings_fantasypoints']]

super_bowl_pred = model.predict(sc.transform(X_superbowl))
print(super_bowl_pred)

sb_predicted = []
for i in super_bowl_pred:
    sb_predicted.append(i[0])
expected = y_superbowl['player_draftkings_fantasypoints'].values

#Plot visualization of expected vs predicted
df3 = pd.DataFrame()
df3['Expected'] = pd.Series(expected)
df3['Predicted'] = pd.Series(sb_predicted)
figure = plt.figure(figsize=(11,11))
axes = sns.scatterplot(data=df3, x='Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)
start=min(expected.min(), super_bowl_pred.min())
end=max(expected.max(), super_bowl_pred.max())
axes.set_xlim(start,end+1)
axes.set_ylim(start,end+1)
line=plt.plot([start,end],[start,end],'k--')

print(f'Mean Absolute Error of the Super Bowl Set: {mean_absolute_error(y_superbowl, super_bowl_pred)}')




