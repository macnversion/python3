# -*- coding: utf-8 -*-
# %%
import codecs
import requests
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import json
import datetime

# %% 数据集
r = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with codecs.open('SIEP3_iris.txt', 'w', encoding='utf-8') as f:
    f.write(r.text)
with codecs.open('SIEP3_iris.txt', 'r', encoding = 'utf-8') as f:
    lines = f.readlines()

for line in lines:
    print(line)

# %% 
irisdata = pd.read_csv('SIEP3_iris.txt', header=None, encoding='utf-8')
cnames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
irisdata.columns = cnames

# 快速过滤
irisdata[irisdata['petal_length']==irisdata.petal_length.max()]

# %% series
series1 = pd.Series(np.random.randn(4))

# %% index
multi1 = pd.Index([('Row_' + str(x+1), 'Col_' + str(y+1)) for x in range(4) for y in range(4)])
multi1.name = ['index1', 'index2']

data_for_multi1 = pd.Series(range(16), index=multi1) # 多重索引的数据变形
data_for_multi1.unstack()

period_index1 = pd.period_range('2018-01', '2018-12', freq='M')
period_index2 = period_index1.asfreq('D', how='start')
period_index3 = period_index1.asfreq('D', how='end')

period_index_mon = pd.period_range('2018-01', '2018-03', freq='M').asfreq('D', how='start')
period_index_day = pd.period_range('2018-01-01', '2018-03-31', freq='D')
full_ts = pd.Series(period_index_mon, index=period_index_mon).reindex(period_index_day, method='ffill')


# %% 测试python语法的语句
class Cars():
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0

    def get_descriptive_name(self):
        pass

    def read_odometer(self):
        print("This car has " + str(self.odometer_reading) + " miles on it.")

    def update_odometer(self, mileage):
        self.odometer_reading = mileage

