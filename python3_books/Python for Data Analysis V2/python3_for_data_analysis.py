# -*- coding: utf-8 -*-
# %%
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os
import platform
from datetime import datetime, date, time
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import json
import csv
# %% dataset路径
'''
value = true-expr if condition else false-expr
'''

mac_path = r'/Users/machuan/CodeSpace/Python/python3'
win_path = r''
workpath = win_path if 'windows' in platform.platform else mac_path
os.chdir(workpath)

# %% 基本数据结构
# 取出元组
tup = (4,5,6)
a, b, c = tup
print('a=', a, 'b=', b, 'c=', 'c')

sequence = [(1,2,3), (4,5,6), (7,8,9)]
for a,b,c in sequence:
    print('a={0}, b={1}, c={2}'.format(a,b,c))
    
values = 1, 2, 3, 4, 5
a, b, *rest = values
print('rest=', rest)

# enumerate
some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, value in enumerate(some_list):
    mapping[i] = value
print('mapping=', mapping, '\n')

# zip 返回包含tuple的list
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1, seq2)
print('zipped=', zipped)
list(zipped)

pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'), 
            ('Schilling', 'Curt')]
first_name, last_name = zip(*pitchers)
print(first_name)
print(last_name)

# dict
words = ['apple', 'bat', 'bar', 'atom', 'book']

by_letter = {}
for word in words: # 实现方法一
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)

by_letter = {}
for word in words: # 使用setdefault实现
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)

by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)

# %% numpy
plt.style.use('ggplot')
m, n = (5,3)
x = np.linspace(0, 1, m)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
plt.plot(X, Y, marker='.', color='blue', linestyle='none')

z = [i for i in zip(X.flat, Y.flat)] # 所有的点

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs**2 + ys**2)
plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.title('image of grid values')

# rnadom walk
# fix me to add a new fig
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
plt.plot(walk)
# %% pandas
df = pd.read_csv('./dataset/ex6.csv')

chunker = pd.read_csv('./dataset/ex6.csv', chunksize=1000)
tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)

data = pd.read_json('./dataset/population_data.json')
# %% 时间序列
dt = datetime(2011, 10, 29, 20, 30, 21)
dt.year
dt.time()