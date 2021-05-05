#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:52:33 2021

@author: machuan
"""
# %% 加载需要的包
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib as plt
import os

# %%
df = pd.read_csv('./dataset/pandas/learn_pandas.csv')
print('df的列名称包含:\n{0}'.format(df.columns))
df = df[df.columns[0:7]]
print('df.shape is:{0}'.format(df.shape))
print('精简后的df的列名称包含:\n{0}'.format(df.columns))

# 一些基础的函数
df.head()
df.tail()
df.info()
df.describe()

df_demo = df[['Height', 'Weight']]
df_demo.max()

df['School'].unique() # 返回去重的结果
df['School'].nunique() # 返回不含重复项的数量
df['School'].value_counts() # 类似groupby后的count计数操作

# 再起构造一个数据
df_demo = df[['Gender', 'Transfer', 'Name']]
df_demo.drop_duplicates(['Gender', 'Transfer'])
df_demo.drop_duplicates(['Gender', 'Transfer'], keep='last')

# 替换函数：映射替换、逻辑替换、数值替换
df['Gender'].replace({'Female':0, 'Male':1})
df['Gender'].replace(['Female', 'Male'], [0, 1])
