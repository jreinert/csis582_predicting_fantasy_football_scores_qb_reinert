# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:54:48 2020

@author: reine
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn import metrics
from sklearn.metrics import mean_absolute_error


pd.set_option('precision', 4)

#load complete data
qb_dataset = pd.read_csv('D:/Documents/NFL Data/qb_data2.csv')

#Create new dataframe containing features
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
                 'player_draftkings_fantasypoints']]

df.describe()

#Create dataframe for X vars
X_data = df[['player_team_score',
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

#Create dataframe for y var
y_data = df[['player_draftkings_fantasypoints']]

#Plot data for visualization
#sns.set_style('whitegrid')
#for feature in df:
#    plt.figure(figsize=(8, 4.5))
#    sns.scatterplot(data=qb_dataset, x=feature,
#                    y='player_draftkings_fantasypoints',
#                    hue='player_draftkings_fantasypoints',
#                    palette='cool', legend=False)


#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,random_state=0,test_size=0.2)

#Train the model
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)
print(f'Intercept: {linear_regression.intercept_}')

for i, name in enumerate(X_data.columns.values.tolist()):
    print(f'{name:>10}: {linear_regression.coef_[0][i]}')
    
#Test the model
pred = linear_regression.predict(X_test)
predicted = []
for i in pred:
    predicted.append(i[0])
expected = y_test['player_draftkings_fantasypoints'].values
   
#Plot visualization of expected vs predicted
df2 = pd.DataFrame()
df2['Expected'] = pd.Series(expected)
df2['Predicted'] = pd.Series(predicted)
figure = plt.figure(figsize=(11,11))
axes = sns.scatterplot(data=df2, x='Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)
start=min(expected.min(), pred.min())
end=max(expected.max(), pred.max())
axes.set_xlim(start,end)
axes.set_ylim(start,end)
line=plt.plot([start,end],[start,end],'k--')

print(metrics.r2_score(expected, predicted))
estimators = {
        'LinearRegression': linear_regression,
        'ElasticNet': ElasticNet(),
        'Lasso': Lasso(),
        'Ridge': Ridge()}

for estimator_name, estimator_object in estimators.items():
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    scores = cross_val_score(estimator=estimator_object, X=X_data,y=y_data,cv=kfold,scoring='r2')
    print(f'{estimator_name:>16}: mean of r2 scores={scores.mean():.3f}')

print(linear_regression.score(X_test, y_test))
print(f'Mean Absolute Error of the Test Set: {mean_absolute_error(y_test, pred)}')

y = 0
count = 1
total = 0
lessthan2 = 0
lessthan5 = 0
for z in expected:
    if predicted[y] >= z:
        total += 1
    if abs(z-predicted[y]) <= 2:
        lessthan2 += 1
    if abs(z-predicted[y]) <= 5:
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
super_bowl_pred = linear_regression.predict(X_superbowl)

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

print(linear_regression.score(X_superbowl, y_superbowl))
print(f'Mean Absolute Error of the Super Bowl Set: {mean_absolute_error(y_superbowl, super_bowl_pred)}')