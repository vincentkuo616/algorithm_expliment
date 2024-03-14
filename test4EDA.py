# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:29:25 2023

@author: vincentkuo
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\vincentkuo\\Downloads\\bulidata.csv\\bulidata.csv', index_col=0 )
#print(df.head())

'''Check Missing Values
'''

# filter columns with empty values
empty_cols = df.columns[df.isna().any()].tolist()

# create separate df only with columns which consist empty values
df_isnull = df[empty_cols]

#print(df_isnull.isnull().sum())
print(df.info())

#Checking for wrong entries like symbols -,?,#,*,etc.
#瀏覽所有 Unique 過後的資料內容
#for col in df.columns:
#    print('{} : {}'.format(col,df[col].unique()))

#sns.heatmap(df.isnull())

'''Replace Missing Value
'''
#plt.figure(figsize=(8,5))
#
#sns.barplot(data=df, x='LOCATION', y='VIEWER')
#
## Rotate the x-axis labels by 90 degrees
#ax = plt.gca()
#ax.tick_params(axis='x', labelrotation=90)

# Try to replace the VIEWER with predicted values by linear regression

# import lr
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# copy df for first try
df_try1 = df.copy()

### timeline fix
df_try1[['DRAW', 'WIN_HOME', 'WIN_AWAY']] = df_try1[['DRAW', 'WIN_HOME', 'WIN_AWAY']].fillna(0)

#prepare train data
df_train = df_try1.select_dtypes(include = np.number).dropna()
train_x = df_train.drop('VIEWER', axis =1)
train_y = df_train['VIEWER']

lr.fit(train_x,train_y)

# prediction into new df
viewer_predicted = pd.DataFrame(lr.predict(df_try1.drop('VIEWER', axis =1).select_dtypes(include = np.number).dropna()), columns=['VIEWER'])
#viewer_predicted['VIEWER'] = pd.DataFrame(lr.predict(df_try1.drop('VIEWER', axis =1).select_dtypes(include = np.number).dropna()))

#use viewer_predicted to replace the origin values
df_try1.VIEWER.fillna(viewer_predicted.VIEWER, inplace = True)

# to have a look on the result
#plt.figure(figsize=(8,5))
#
#sns.barplot(data=df_try1, x='LOCATION', y='VIEWER')
#
## Rotate the x-axis labels by 90 degrees
#ax = plt.gca()
#ax.tick_params(axis='x', labelrotation=90)

# Next Try to replace the NaN values based on the mean per city and if the city has no real entry I will set the viewer to 0 .
# copy for try
df_try2 = df.copy() 

# creating dictionary with means per location 
df_grouped = df.groupby('LOCATION')['VIEWER'].mean().dropna().reset_index()
location_mean = df_grouped.set_index('LOCATION')['VIEWER'].to_dict()

# map dictionary
df_try2['VIEWER'] = df_try2['LOCATION'].map(location_mean)

# fill unknown stadiums with zero
df_try2.fillna(0, inplace= True)

# to have a look on the result
#plt.figure(figsize=(8,5))
#
#sns.barplot(data = df_try2, x= 'LOCATION', y = 'VIEWER')
#
## Rotate the x-axis labels by 90 degrees
#ax = plt.gca()
#ax.tick_params(axis='x', labelrotation=90)

df = df_try2

# filter columns with empty values
empty_cols = df.columns[df.isna().any()].tolist()

# create separate df only with columns which consist empty values
df_isnull = df[empty_cols]

print(df_isnull.isnull().sum())

'''Checking Dublicates
'''

# filtering on duplicates
duplicates = df[df.duplicated(keep=False)]

# show duplicates 
#print(duplicates)

#if necessary .drop_duplicates()

'''EDA
'''
# Numarical Vars
#print(df.columns)
#print(df.describe().T)

numarical = df[['GOALS_HOME', 'GOALS_AWAY','VIEWER']]
print(numarical.sum())

#fig, ax = plt.subplots(3,2, figsize=(10,10))
#
#for i, var in enumerate(numarical.columns):    
#    if i <=1:
#        sns.histplot(data=numarical, x=var, ax=ax[i,0], element='bars', discrete=True)
#    else:
#        sns.histplot(data=numarical, x=var, ax=ax[i,0], element='bars', kde= True)
#    sns.boxenplot(numarical, x=var, ax=ax[i,1])

# Categorical Vars
variables = ['MATCH_DATE', 'LEAGUE_NAME', 'LOCATION','MATCHDAY', 'SEASON', 'LEAGUE', 'FINISHED', 
     'MATCHDAY_NR', 'DRAW', 'WIN_HOME', 'WIN_AWAY']
categorical = df[variables]

print(categorical.head())

#fig, ax = plt.subplots(11,1, figsize=(10,50))
#
#for i, var in enumerate(categorical.columns):
#    sns.countplot(data=categorical, x=var, ax=ax[i])
#
#    if i<1:
#        ax[i].set_xticklabels('')
#    elif i<4:
#        ax[i].tick_params(axis='x', rotation=45)
#
#    ax[i].set_title(f'Bar Plot of {var}')
#    ax[i].set_xlabel(var)
#    ax[i].set_ylabel('Count')
#
#plt.tight_layout()
#plt.show()

# Correlation Matrix

#sns.heatmap(df.corr().round(2), annot= True, linewidth=.5)

# Selected Analysis Points

# Color dictionary
colors = {
    'BVB': '#FDE100',
    'Bayern': '#DC052D',
    'Leipzig': '#0C2043',
    'Union Berlin': '#FCEA10',
    'Freiburg': '#000000',
    'Leverkusen': '#E32221',
    'Wolfsburg': '#65B32E',
    'Frankfurt': '#E1000F',
    'Mainz': '#ED1C24',
    'Köln': '#ED1C24',
    'Gladbach': '#000000',
    'Bremen': '#1D9053',
    'Hoffenheim': '#1C63B7',
    'Augsburg': '#BA3733',
    'Stuttgart': '#E32219',
    'Bochum': '#005CA9',
    'Schalke': '#004D9D',
    'Hertha': '#004D9E'
}

current_season = df[df.SEASON == 2022]
#print(current_season.head())

# Whole Season
table_home = current_season.groupby('HOME_TEAM').agg({'DRAW':'sum',
                                                 'WIN_HOME':'sum',
                                                 'GOALS_HOME':'sum',
                                                 'GOALS_AWAY':'sum',}).reset_index()
table_away = current_season.groupby('AWAY_TEAM').agg({'DRAW':'sum',
                                                 'WIN_AWAY':'sum',
                                                 'GOALS_HOME':'sum',
                                                 'GOALS_AWAY':'sum',}).reset_index()

table_home = table_home.rename(columns={'HOME_TEAM':'TEAM',
                                        'WIN_HOME':'WINS',
                                        'GOALS_HOME':'GOALS',
                                        'GOALS_AWAY':'GOALS_CONCEDED'})
table_home['POINTS'] = (table_home['WINS']*3) + table_home['DRAW']
table_home['LOSES'] = 17 - table_home['WINS'] - table_home['DRAW']
table_home['DIFFERENCE'] = table_home['GOALS'] - table_home['GOALS_CONCEDED']

table_away = table_away.rename(columns={'AWAY_TEAM':'TEAM',
                                        'WIN_AWAY':'WINS',
                                        'GOALS_HOME':'GOALS_CONCEDED',
                                        'GOALS_AWAY':'GOALS'})
table_away['POINTS'] = (table_away['WINS']*3) + table_away['DRAW']
table_away['LOSES'] = 17 - table_away['WINS'] - table_away['DRAW']
table_away['DIFFERENCE'] = table_away['GOALS'] - table_away['GOALS_CONCEDED']

order = ['TEAM', 'POINTS', 'WINS', 'DRAW', 'LOSES', 'GOALS', 'GOALS_CONCEDED', 'DIFFERENCE']
table_away = table_away[order]
table_home = table_home[order]

table = pd.concat([table_home,table_away]).reset_index(drop=True)

table = table.groupby("TEAM", as_index=False).sum().sort_values(by= ['POINTS', 'DIFFERENCE'], ascending=False).reset_index(drop=True)

print(table)

vars = ['POINTS', 'WINS', 'DRAW', 'LOSES', 'GOALS', 'GOALS_CONCEDED', 'DIFFERENCE']

#fig, ax = plt.subplots(7,1, figsize=(8,40))
#for i, var in enumerate(vars):
#    table.sort_values(by=[var, 'POINTS', 'DIFFERENCE'], ascending=False, inplace= True)
#    sns.barplot(data=table, x=var, y='TEAM', palette=colors, ax= ax[i])
#
#    for j, points in enumerate(table[var]):
#        ax[i].text(points, j, str(points), color='black', va='center')
#
#    ax[i].set_xlabel(var)
#    ax[i].set_ylabel('Team')
#
#plt.show()

# Home Table

table_home.sort_values(by= ['POINTS', 'DIFFERENCE'], ascending=False, inplace=True)
table_home.reset_index(drop=True , inplace=True)
print(table_home)

#fig, ax = plt.subplots(7,1, figsize=(8,40))
#for i, var in enumerate(vars):
#    table_home.sort_values(by=[var, 'POINTS', 'DIFFERENCE'], ascending=False, inplace= True)
#    sns.barplot(data=table_home, x=var, y='TEAM', palette=colors, ax= ax[i])
#
#    for j, points in enumerate(table_home[var]):
#        ax[i].text(points, j, str(points), color='black', va='center')
#
#    ax[i].set_xlabel(var + ' home')
#    ax[i].set_ylabel('Team')
#
#plt.show()

# Away Table

table_away.sort_values(by= ['POINTS', 'DIFFERENCE'], ascending=False, inplace=True)
table_away.reset_index(drop=True , inplace=True)
print(table_away)

#fig, ax = plt.subplots(7,1, figsize=(8,40))
#for i, var in enumerate(vars):
#    table_away.sort_values(by=[var, 'POINTS', 'DIFFERENCE'], ascending=False, inplace= True)
#    sns.barplot(data=table_away, x=var, y='TEAM', palette=colors, ax= ax[i])
#
#    for j, points in enumerate(table_away[var]):
#        ax[i].text(points, j, str(points), color='black', va='center')
#
#    ax[i].set_xlabel(var + ' away')
#    ax[i].set_ylabel('Team')
#
#plt.show()

current_season['RESULT'] = current_season['GOALS_HOME'].astype(str) + ' : ' + current_season['GOALS_AWAY'].astype(str)

results = current_season[['RESULT', 'HOME_TEAM', 'AWAY_TEAM']].reset_index(drop=True)

pivot_results = results.pivot(index= 'HOME_TEAM',columns='AWAY_TEAM', values='RESULT')

pivot_results_copy = pivot_results.copy()

pivot_results_copy = pivot_results_copy.applymap(lambda x: 1)

fig, ax = plt.subplots(1,1, figsize=(15,15))

sns.heatmap(pivot_results_copy, annot=pivot_results.values, fmt="", linewidth=3, cbar=False, cmap='YlGnBu')


