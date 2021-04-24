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
df = df[df.columns[0:6]]
print('df.shape is:{0}'.format(df.shape))
print('精简后的df的列名称包含:\n{0}'.format(df.columns))
