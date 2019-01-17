# -*- coding: utf-8 -*-
# %%
import codecs
import requests
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import json

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