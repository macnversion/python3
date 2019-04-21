# -*- coding: utf-8 -*

# %%
import os
import platform
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# %%
#base = importr('base')
#tidyverse = importr('tidyverse')

# %% 计算代码的执行时间
# %timeit L = [n ** 2 for n in range(100)]

# %% numpy
# numpy中的可以直接应用的数组
np.zeros(10, dtype=int) # 创建长度为10的零数组
np.ones((3, 5), dtype=float) # 创建3✖*5的浮点数组
np.full((3, 5), 3.14) # 创建3*5的浮点数组，数值全部都是3.14
np.arange(0, 20, 2) # 0开始，20结束，步长为2
np.linspace(0, 1, 5) # 0到1之间均匀分布的5个数值
np.random.randint(0, 10, (3, 3)) # 创建一个3*3的随机数组
np.eye(3)

np.random.seed(0)
x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))
print('x3 ndim:', x3.ndim, '\n')
print('x3 shape:', x3.shape, '\n')
print('x3 size:', x3.size, '\n')
print('x3 dtype:', x3.dtype, '\n')

'''
numpy中的数组是固定类型的，插入一个浮点数到整型的数组中，浮点数将会被截短成为整型
'''
print('x1=', x1)
x1[0] = 3.14
print('x1=', x1)

# 获取短数组： x[start: stop: step]
# 数组的变形
x = np.array([1, 2, 3])
x.reshape(1,3)
x[np.newaxis, :] # 通过np.newaxis变形
x[:, np.newaxis]

# 数组的拼接和分裂
'''
数组拼接函数
np.concatenate([x, y, z])
np.vstack([x, grid])
np.hstack([x, grid])
'''
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = np.array([99, 99, 99])
np.concatenate([x, y])
np.concatenate([x, y, z])

grid = np.array([[1, 2, 3],
                [4, 5, 6]])
np.concatenate([grid, grid])
np.concatenate([grid, grid], axis=1)

np.vstack([x, grid]) # 垂直方向的合并
y = np.array([[99],
              [99]])
np.hstack([y, grid]) # 水平方向的合并

'''
数组分裂的函数
np.split(x, index)
np.vsplit()
np.hsplit()
'''
x = [1, 2, 3, 4, 99, 99, 3, 2, 1]
grid = np.arange(16).reshape((4, 4))

# 通用函数
def compute_reciprocals(values):
    out = np.empty(len(values))
    for i in range(len(values)):
        out[i] = 1.0/values[i]
    return out


values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)
# big_array = np.random.randint(1, 100, size=1000000)
# %timeit compute_reciprocals(big_array) # run this in console

# 通用函数的高级特性
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y) # out参数指定输出的位置


# 计算美国总统的身高
data = pd.read_csv('./dataset/python数据科学手册/president_heights.csv')
heights = np.array(data['height(cm)'])
print('mean height:', heights.mean())
print('standard deviation:', heights.std())
print('min height:', heights.min())
plt.hist(heights)

# 聚合
M = np.random.random((3, 4))
M.min(axis=0)

# 广播
X = np.random.random((10, 3))
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# 布尔掩码
rainfall_full_data = pd.read_csv('./dataset/python数据科学手册/Seattle2014.csv')
rainfall = rainfall_full_data['PRCP'].values
inches = rainfall/254
sns.set()
plt.hist(inches, 40)
rainy = (inches > 0)

# 花式索引
rand = np.random.RandomState(42)
x = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
x[row[:, np.newaxis], col]

    
# 数组区间的划分
np.random.seed(42)
x = np.random.randn(100)
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)
i = np.searchsorted(bins, x) # 这是一个神奇的函数
np.add.at(counts, i, 1)


# %% 数组的排序
def selection_sort(x): # 选择排序
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return(x)



x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3) # 返回x第3小的数值，两个分隔区内的数值都是任意排列的

# 利用argsort寻找最邻近
X = np.random.rand(10, 2)
plt.scatter(X[:, 0], X[:, 1])
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=-1)

# 结构化数组
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
data['name'] = name
data['age'] = age
data['weight'] = weight
print('data type is', data.dtype)
print('data is', data)


# %% pandas
area = pd.Series({'Alaska':1723337, 'Texas':695662,'California':423967, 'Chinna':1213345, 'LinnXi':1234346},
                 name='area')
population = pd.Series({'California':38332521, 'Texas':26448193, 'New York': 19651127, 'Chinna':441213,
                        'LinnXi':3451234}, name='population')
data = pd.DataFrame({'area':area, 'population':population})
data['density'] = data['population']/data['area']
data['id'] = np.random.randint(10, size=6)
data.iloc[:3, :2] # 使用iloc的隐式索引
data.loc[:'New York', :'population'] # 使用loc的显式索引
#data.ix[:3, :'population'] # ix混合索引，ix同样存在容易混淆的问题, .ix is deprecated
data.loc[data.density > 90, ['population', 'density']]

# 缺失值
df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]],
    columns = list('ABC'))

'''
axis=0, 输出的结果保留完整的列标签
axis=1，输出的结果保留完整的行标签
'''
df.dropna(axis=0) # axis=0, 输出的结果保留完整的列标签, dropna默认保留列标签
df.dropna(axis=1) # axis=1, 输出的结果保留完整的行表钱

df.fillna(method='ffill', axis=1)


# 层次索引
index1 = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index1) # 使用tuple建立的索引应用起来不方便

index_mul = pd.MultiIndex.from_tuples(index1)
pop = pop.reindex(index_mul)
pop.index.names = ['state', 'year']

pop_df = pop.unstack()
pop_df.stack()

index2 = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names = ['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names = ['subject', 'type'])
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index2, columns=columns)

'''
前面的例子中使用的MultiIndex都是按照字典顺序已经完成了排列的。
对于不是使用字典顺序排列的index，切片索引会出错。
'''
index3 = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.randn(len(index3)), index=index3)
try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)

data = data.sort_index()

'''索引的设置和重置'''
pop_flat = pop.reset_index(name='population')
pop_flat.set_index(['state', 'year'])


'''多级索引的数据累计方法'''
health_data_mean = health_data.mean(level='year')
health_data.mean(axis=0, level='visit')

'''数据集的合并操作'''
def make_df(cols, ind): # 构建一个简单的dtaFrame
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return DataFrame(data, index=ind)

df1 = make_df('AB', [1,2])
df2 = make_df('AB', [3,4])
pd.concat([df1, df2])
pd.concat([df1, df2], axis=1)

'''捕捉索引重复的错误'''
df2.index = df1.index
try:
    pd.concat([df1, df2], verify_integrity=True)
except ValueError as e:
    print('ValueError:', e)

'''忽略索引'''
pd.concat([df1, df2], ignore_index=True)

'''增加多级索引,也可以处理索引重复的问题'''
pd.concat([df1, df2], keys=['x', 'y'])

df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
pd.concat([df5, df6], join='inner')

'''合并数据集'''
df1 = DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                 'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = DataFrame({'name': ['Lisa', 'Bob', 'Jake', 'Sue'],
                 'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.merge(df1, df2, left_on='employee', right_on='name')
df4 = DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                 'supervisor': ['Carly', 'Guido', 'Steve']})
pd.merge(df3, df4)
df5 = DataFrame({'group': ['Accounting', 'Accounting', 'Engineering', 'Engineering', 'HR', 'HR'],
                 'skills': ['math', 'spreadsheets', 'coding', 'linux', 'spreadsheets', 'organization']})
pd.merge(df1, df5)

print('\ndf1=\n', df1, '\n\ndf2=\n', df2, '\ndf3=\n', df3, '\ndf4=\n', df4, '\ndf5=\n', df5)
'''删除多余的列'''
pd.merge(df1, df2, left_on='employee', right_on='name').drop('name', axis=1)
'''索引的合并'''
df1a = df1.set_index('employee')
df2a = df2.set_index('name')
pd.merge(df1a, df2a, left_index=True, right_index=True)

'''数据合并的集合操作规则'''
df6 = DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                 'food': ['fish', 'beans', 'bread']}, columns=['name', 'food'])
df7 = DataFrame({'name': ['Mary', 'Joseph'],
                 'drink': ['wine', 'beer']}, columns=['name', 'drink'])
pd.merge(df6, df7, how='inner')
# %% 示例：美国各州的统计数据
pop = pd.read_csv('./dataset/python数据科学手册/state-population.csv')
areas = pd.read_csv('./dataset/python数据科学手册/state-areas.csv')
abbrevs = pd.read_csv('./dataset/python数据科学手册/state-abbrevs.csv')
merged = pd.merge(pop, abbrevs, how='outer', left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', axis=1)
print('head of merged is:\n', merged.head())
merged.isnull().any() # 检查列数据是否存在缺失值
merged[merged['population'].isnull()].head()
merged.loc[merged['state'].isnull(), 'state/region'].unique()
merged.loc[merged['state/region']=='PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region']=='USA', 'state'] = 'United States'
merged.isnull().any()
final = pd.merge(merged, areas, on='state', how='left')
print('\nhead of final merged data is:\n', final.head())
final.isnull().any() # 面积的数据也存在缺失的
final.loc[final['area (sq. mi)'].isnull(), 'state'].unique()
final.loc[final['state']=='United States', 'area (sq. mi)'] = areas['area (sq. mi)'].sum()
data2010 = final.query('year==2010 & ages=="total"')
data2010.set_index(['state'], inplace=True)
density = data2010['population']/data2010['area (sq. mi)']
density.sort_values(ascending=False, inplace=True)
# %% 数据的累计与分组, split-apply-combine
planets = sns.load_dataset('planets')
planets.isnull().any() # 检查数据是否存在缺失
planets.dropna().describe()
for (method, group) in planets.groupby('method'):
    print('{0:30s} shape={1}'.format(method, group.shape))

planets.groupby('method').aggregate(['min', np.median, max])
planets.groupby('method').aggregate({'mass':max, 'year':min})

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
    columns=['key', 'data1', 'data2'])


def filter_func(x):
    return x['data2'].std() > 4


df.groupby('key').filter(filter_func)
df.groupby('key').transform(lambda x: x - x.mean())
L = [0, 1, 0, 1, 2, 0]
df.groupby(L).sum()
df2 = df.set_index('key')
mapping = {'A':'vowel', 'B':'consonant', 'C':'consonant'}
df2.groupby(mapping).sum() # 将索引映射到分组名称
df2.groupby(str.lower).mean() # 将索引映射到任意的python函数
df2.groupby([str.lower, mapping]).mean()

decade = 10*(planets['year']//10)
decade = decade.astype(str) + 's'
decade.name = decade
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

# %% 数据透视表
titanic = sns.load_dataset('titanic')
titanic.groupby(['sex', 'class'])['survived'].mean().unstack() # 使用groupby函数生成分析数据
titanic.pivot_table('survived', index='sex', columns='class', aggfunc='mean')
age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', index=['sex', age], columns='class', aggfunc='mean')
fare = pd.qcut(titanic['fare'], 2)
titanic.pivot_table('survived', index=['sex', age], columns=['class', fare], aggfunc='mean')

births = pd.read_csv('./dataset/python数据科学手册/births.csv')
births['decade'] = 10*(births['year']//10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
sns.set() # 使用seaborn风格
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()

# %% 向量化字符串操作
try:
    recipes = pd.read_json('./dataset/python数据科学手册/recipeitems-latest.json')
except ValueError as e:
    print('ValueError:', e)

'''
检查github上的数据的获取链接
'''
# %% 时间序列

# %% matplotlib 
x = np.linspace(0, 10000, 100)
fig = plt.figure()
ax = plt.axes()
'''使用ax.set一次设置所有的属性'''
ax.set(xlim=(0, 10000), ylim=(-2, 2),
       xlabel='x', ylabel='sin(x) and cos(x)')
plt.plot(x, np.sin(x), '-r', label='sin(x)')
plt.plot(x, np.cos(x), '--g', label='cos(x)')
plt.title('a sin and cos curve')
plt.legend()
# %%
fig2 =plt.figure()
ax = plt.axes()
rng = np.random.RandomState(0)
for marker in ['o','.', ',', 'x', '+', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker, label="maker='{0}'".format(marker))
plt.lengend(numpoints=1)

# %%
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], alpha=0.2, s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
# %%
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy*np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.r')