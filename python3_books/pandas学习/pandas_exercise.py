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
