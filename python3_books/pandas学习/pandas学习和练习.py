#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:52:33 2021

@author: machuan
pandas常用的75个操作
"""
# %% 加载需要的包
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib as plt
import os
import string

# %% 查看pandas的版本
print(pd.__version__)
print(pd.show_versions())

# %% 使用python list、python dict、numpy.ndarray创建pandas.series
mylist = list(string.ascii_lowercase) # python list
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))

ser1 = pd.Series(mylist)
ser2 = pd.Series(myarr)
ser3 = pd.Series(mydict)

print(f'ser1 is \n{ser1}')
print(f'ser2 is \n{ser2}')
print(f'ser3 is \n{ser3}')