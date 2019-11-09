# -*- coding: utf-8 -*

import numpy as np
import pandas as pd

path1 = './python3/dataset/pandas_exercise_data/chipotle.tsv'
chipo = pd.read_csv(path1, sep='\t')

c = chipo[['item_name', 'quantity']].groupby(['item_name'], as_index=False).agg({'quantity':sum})
c.sort_values(['quantity'], ascending=False, inplace=True)
c.head()

chipo['item_name'].unique().shape
chipo['item_name'].nunique()
chipo['choice_description'].value_counts().head(5)


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


path3 = './python3/dataset/pandas_exercise_data/drinks.csv'
drinks = pd.read_csv(path3)
drinks.groupby('continent')['beer_servings'].mean()
drinks.groupby('continent')['wine_servings'].describe()


path4 = './python3/dataset/pandas_exercise_data/US_Crime_Rates_1960_2014.csv'
crime = pd.read_csv(path4)
crime['Year'] = pd.to_datetime(crime['Year'], format='%Y')
crime = crime.set_index()