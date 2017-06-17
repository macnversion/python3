# -*- coding: utf-8 -*-

# %%
import platform
import numpy as np
import matplotlib as plt
import jinja2
import pandas as pd
import csv

# %% 数据路径设置
windows_path = r'D:/WorkSpace/CodeSpace/Code.Data/R'
mac_path = r'/Users/machuan/CodeSpace/Code.Data/R'
data_path = windows_path if platform.system()=='Windows' else mac_path

# %% ch06 运用税务数据进行应用导向的数据分析
ch_data_path = data_path + r'/数据科学实战手册_R_Python/Chapter06/data'
income_dist_file_path = ch_data_path + '/income_dist.csv'
#income_dist = pd.read_csv(income_dist_file_path)
with open(income_dist_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)

