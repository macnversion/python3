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

# %% 查看pandas的版本
print(pd.__version__)
print(pd.show_versions())

# %% 使用python list、python dict、numpy.ndarray创建pandas.series
my