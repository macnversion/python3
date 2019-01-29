# -*- coding: utf-8 -*-
# %%
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os
import platform
import json
from datetime import datetime, date, time
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import requests
import re
import seaborn
from dateutil.parser import parse
from collections import defaultdict
from collections import Counter

# %% dataset路径
'''
value = true-expr if condition else false-expr
'''

#mac_path = r'/Users/machuan/CodeSpace/Python/python3'
#win_path = r''
#workpath = win_path if 'windows' in platform.platform else mac_path
#os.chdir(workpath)

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

# %% 数据去重
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                     'k2': [1, 1, 2, 3, 3, 4, 4]})
data.duplicated()
data.drop_duplicates()
# %% 数据透视
data = pd.read_csv('./dataset/macrodata.csv')
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
data = data.reindex(columns=columns)
data.index = periods.to_timestamp('D', 'end')
ldata = data.stack().reset_index().rename(columns={0: 'value'})
pivoted = ldata.pivot('date', 'item', 'value')

ldata['value2'] = np.random.randn(len(data))
pivoted = ldata.pivot('data', 'item')
# %%
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'Pastrami', 'corned beef', 'Bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_animal = {
    'bacon': 'pig',
    'pulled pork': 'pig',
    'pastrami': 'cow',
    'corned beef': 'cow',
    'honey ham': 'pig',
    'nova lox': 'salmon'
}
data['animal'] = data['food'].str.lower().map(meat_to_animal)
data['animal'] = data['food'].map(lambda x: meat_to_animal[x.lower()])

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
cats = pd.cut(ages, bins, labels=group_names)
# %% 数据汇总和分组计算
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'], 
                   'data1' : np.random.randn(5), 
                   'data2' : np.random.randn(5)})

# %% API相关交互
url = r'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)
data = resp.json()
issues = DataFrame(data, columns=['number', 'title', 'labels', 'states'])
# %% 字符串处理 & 正则表达式
val = 'a,b, guido'
val.split(',')
pieces = [x.strip() for x in val.split(',')]

text = "foo    bar\t baz  \tqux"
re.split('\s+', text)

regex = re.compile('\s+')
regex.findall(text)
# %% 时间序列
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
         datetime(2011, 1, 7), datetime(2011, 1, 8), 
         datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
ts['20110102'] # 可直接使用字符串索引

long_ts = Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
long_ts['2001']
long_ts.truncate(after='1/9/2001')

# %% 数据可视化
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
plt.plot(np.random.randn(50).cumsum(), 'k--')
_ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30)+3*np.random.randn(30))

# %% 线图
fig, axes = plt.subplots(2,1)
s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
                  columns=['A', 'B', 'C', 'D'],
                  index=np.arange(0, 100, 10))
axes[0].plot(s)
axes[1].plot(df)
# %% 条形图
fig, axes = plt.subplots(2,1)
data = Series(np.random.randn(16), index=list('abcdefghijklmnop'))
data.plot.bar(ax=axes[0], color='k', alpha=0.7)
data.plot.barh(ax=axes[1], color='k', alpha=0.7)

df = pd.DataFrame(np.random.rand(6, 4),
                  index=['one', 'two', 'three', 'four', 'five', 'six'],
                  columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df.plot.bar()

# %% 柱状图和密度图
tips = pd.read_csv('./dataset/tips.csv')
party_counts = pd.crosstab(tips['day'], tips['size'])
party_counts = party_counts.loc[:, 2:5]
party_pcts = party_counts.div(party_counts.sum(1), axis=0)


# %% usagov数据的应用
data_path = r'./dataset/bitly_usagov/example.txt'
with open(data_path) as f:
    data = f.readlines()

records1 = [json.loads(line) for line in open(data_path)]
records = [json.loads(line) for line in data]

time_zones = [rec['tz'] for rec in records if 'tz' in rec]

# 使用python对时区进行计数
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


def get_counts2(sequence):
    counts = defaultdict(int) # 所有的值均会初始化为0
    for x in sequence:
        counts[x] += 1
    return counts


counts = get_counts(time_zones)


def top_counts(counts_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in counts_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

counts = Counter(time_zones)
counts.most_common(10)


# 使用pandas对时区进行计数
df_records = DataFrame(records)
tz_counts = df_records['tz'].value_counts()
clean_tz = df_records['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
