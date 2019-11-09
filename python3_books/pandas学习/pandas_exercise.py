# -*- coding: utf-8 -*
# %%
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling

# %%
path1 = './python3/dataset/pandas_exercise_data/chipotle.tsv'
chipo = pd.read_csv(path1, sep='\t')

c = chipo[['item_name', 'quantity']].groupby(['item_name'], as_index=False).agg({'quantity':sum})
c.sort_values(['quantity'], ascending=False, inplace=True)
c.head()

chipo['item_name'].unique().shape
chipo['item_name'].nunique()
chipo['choice_description'].value_counts().head(5)


# %%
path2 = './python3/dataset/pandas_exercise_data/Euro2012_stats.csv'
euro12 = pd.read_csv(path2)
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=False)
round(discipline['Yellow Cards'].mean())
euro12[euro12['Goals']>6]
euro12[euro12.Team.str.startswith('G')]
euro12.iloc[:, 0:7]
euro12.iloc[:, 0:-3]
euro12.loc[euro12.Team.isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]


# %%
path3 = './python3/dataset/pandas_exercise_data/drinks.csv'
drinks = pd.read_csv(path3)
drinks.groupby('continent')['beer_servings'].mean()
drinks.groupby('continent')['wine_servings'].describe()

# %%
path4 = './python3/dataset/pandas_exercise_data/US_Crime_Rates_1960_2014.csv'
crime = pd.read_csv(path4)
crime['Year'] = pd.to_datetime(crime['Year'], format='%Y')
crime = crime.set_index('Year', drop=True)

# %%
raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
data1 = pd.DataFrame(raw_data_1, columns=['subject_id', 'first_name', 'last_name'])
data2 = pd.DataFrame(raw_data_2, columns=['subject_id', 'first_name', 'last_name'])
data3 = pd.DataFrame(raw_data_3, columns=['subject_id', 'test_id'])
all_data = pd.concat([data1, data2])
all_data_col = pd.concat([data1, data2], axis=1)


# %% 风速
path6 = './python3/dataset/pandas_exercise_data/wind.data'
data = pd.read_table(path6, sep='\s+', parse_dates=[[0,1,2,]])


def fix_century(x):
    year = x.year - 100 if x.year > 1989 else x.year
    return datetime.date(year, x.month, x.day)


data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_century)
data['Yr_Mo_Dy'] = pd.to_datetime(data['Yr_Mo_Dy'])
data = data.set_index('Yr_Mo_Dy')
# 计算每一个location，一月份的平均风速
data['date'] = data.index
data['month'] = data['date'].apply(lambda x: x.month)
data['year'] = data['date'].apply(lambda x: x.year)
data['day'] = data['date'].apply(lambda x: x.day)
january_winds = data.query('month==1')


# %% 泰坦尼克
path7 = './python3/dataset/pandas_exercise_data/train.csv'
titanic = pd.read_csv(path7)
titanic.set_index('PassengerId')

# %%
path9 = './python3/dataset/pandas_exercise_data/Apple_stock.csv'
apple = pd.read_csv(path9)
apple['Date'] = pd.to_datetime(apple['Date'])
apple = apple.set_index('Date')
apple.sort_index(ascending=True)
pandas_profiling.ProfileReport(apple)