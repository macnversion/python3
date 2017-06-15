# -*- coding: utf-8 -*-

# %%
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt 
import codecs
import requests
import scipy as sp
from datetime import datetime
from dateutil import parser
import json
import seaborn as sns
# %% 
# Accessing Documentation with ?
# Accessing source code with ??
help(len)
L = [1, 2, 3]
# L?
# help??

# %xmode Plain

# %% numpy
L = list(range(10))
L2 = [str(c) for c in L]

np.zeros(10, dtype=int)
np.ones((3, 5), dtype=float)
np.full((3,5), 3.14)
np.arange(0, 20, 2)
np.linspace(0, 1, 5)
np.random.random((3,3))
np.random.normal(0, 1, (3,3))
np.random.randint(0,10,(3,3))
np.eye(3)

x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3,4))
x3 = np.random.randint(10, size=(3,4,5))
print('x3 ndim', x3.ndim)
print('x3.shape', x3.shape)
print('x3.size', x3.size)

# np.concatenate, np.vstack, np.hstack
x = np.array([1,2,3])
y = np.array([4,5,6])
z = np.array([99,99,99])
np.concatenate([x,y,z])

grid = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
print('axis=0:\n', np.concatenate([grid,grid]),'\n')
print('axis=1:\n', np.concatenate([grid,grid], axis=1),'\n')
print('vstack:\n', np.vstack([x,grid]),'\n')
print('hstack:\n', np.hstack([x.reshape(3,1),grid]),'\n')

# np.split, np.vsplit, np.hsplit
x = np.arange(10)
y = np.arange(16).reshape(4,4)
x1, x2, x3 = np.split(x, [3,5])
upper, lower = np.vsplit(y, [2])
left, right = np.hsplit(y, [2])

# advanced ufunc feathers

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 2, out=y) # out参数直接将结果输出到y

# Aggregates
# reduce 将array reduce为一个值
# accumulate 可以保存reduce过程中的每个中间结果
x = np.arange(10)
np.add.reduce(x)
np.multiply.reduce(x)
np.add.accumulate(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)
#plt.imshow(z)
#plt.colorbar()

# %% pandas
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')

population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
data = pd.DataFrame({'area':area, 'population':population})
population = Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
data = DataFrame({'area':area, 'population':population})
data['density'] = data['population']/data['area']
print(data.loc[:'New York', :'population'], '\n')
print(data.iloc[:3,:2])

a = DataFrame(np.random.randint(0, 10, (4,4)), columns=list('abcd'))
b = DataFrame(np.random.randint(0, 100, (5,6)),
              columns=list('abcdef'),
              index=list('abcde'))
c = Series(np.random.random(10), index=list('1234567890'))
# b.ix 可以按照index名称索引，可以按照index行的序号索引
# b.loc 仅能按照index名称索引
# b.iloc 仅能按照index的行的序号索引


# dropping null values
data.isnull()
data.notnull()
data.dropna(how='all')
data.dropna(how='any')
data.dropna(thresh=3) # thresh参数表示值不为nan的个数
data.dropna(axis=1, thresh=3)

# filling mull values
data.fillna(method='ffill')
data.fillna(method='bfill')
data.fillna(method='ffill', axis=1)

# multi-indexing data
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
multi_index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(multi_index)
pop.df = pop.unstack()
pop.df.stack()


pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
pop_df['rate'] = pop_df['under18']/pop_df['total']*100
f_u18 = pop_df['rate']
f_u18.unstack()

pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
pd.MultiIndex.from_product([['a','b'], [0, 1]])
pop.index.names = ['state', 'year']

index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'],
                                      ['HR', 'Temp']],
                                     names=['subject', 'type'])
data = np.round(np.random.randn(4,6),2)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)

index = pd.MultiIndex.from_product([['a', 'b', 'c'], [1, 2]])
data = pd.Series(np.random.randn(6), index=index)
data.index.names = ['char', 'int']

# %% combing datasets
# 构造数据集和加载数据集
def make_df(col, ind):
    data = {c:[str(i)+str(c) for i in ind] for c in col}
    return pd.DataFrame(data, index=ind)
# 可查看sns.load_dataset?
planets = sns.load_dataset('planets')
titanic = sns.load_dataset('titanic')

# %% Aggregation and groupby
planets['mass'].sum()
planets.groupby('method')['orbital_period']
planets.groupby('method')['orbital_period'].median()

for (method, group) in planets.groupby('method'):
    print('{0:30s} shape={1}'.format(method, group.shape))

planets.groupby('method')['year'].describe().stack()

# aggregate
planets.groupby('method').aggregate(['min', np.median, max])
planets.groupby('method').aggregate({'mass':min, 'distance':max})


decade = 10*(planets['year']//10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

# %% pivot tables
titanic.groupby('sex')['survived'].mean()
titanic.groupby(['sex', 'class'])['survived'].mean().unstack()

titanic.pivot_table('survived', index='sex', columns='class')
titanic.pivot_table('survived', ['sex', 'who'], 'class')

age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')

#DataFrame.pivot_table(data, values=None, index=None, columns=None,
#aggfunc='mean', fill_value=None, margins=False,
#dropna=True, margins_name='All')

titanic.pivot_table(index='sex',
                    columns='class',
                    aggfunc={'survived':sum, 'fare':np.mean})

# %% time series


# %% pandas 快速上手
r = requests.get("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
with codecs.open('S1EP3_Iris.txt','w',encoding='utf-8') as f:
    f.write(r.text)
    
with codecs.open('S1EP3_Iris.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    print(line)

iris = pd.read_csv('S1EP3_Iris.txt',header = None, encoding='utf-8')
cnames = ['sepal_length','sepal_width','petal_length','petal_width','class']
iris.columns = cnames
# %%
# 快速过滤
iris[iris['petal_width'] == iris.petal_width.max()]
# 快速切片
iris.iloc[::30, :2]
# 快速统计
iris['class'].value_counts()
# 快速mapreduce
slogs = lambda x: sp.log(x)*x
entpy = lambda x: sp.exp((slogs(x.sum())-x.map(slogs).sum())/x.sum())
iris.groupby('class').agg(entpy)

# %%
multi1 = [('Row_'+str(x+1), 'Col_'+str(y+1)) for x in np.arange(4) \
          for y in np.arange(4)]
multi1 = pd.Index(multi1)
multi1.names = ['index1', 'index2']

dates = [datetime.datetime(2015,1,1),datetime.datetime(2015,1,8),
         datetime.datetime(2015,1,30),datetime.datetime(2015,2,24)]
pd.DatetimeIndex(dates)

periodIndex1 = pd.period_range('2017-01', '2017-05', freq='M')
periodindex_mon = pd.period_range('2017-01', '2017-05', freq='M')\
.asfreq('D', how='start')
periodindex_day = pd.period_range('2017-01', '2017-05', freq='D')

full_ts = pd.Series(periodindex_mon, index=periodindex_mon).reindex(\
                   periodindex_day, method='ffill')

# %% 半结构化数据
json_data = [{'name':'Wang','sal':50000,'job':'VP'},\
{'name':'Zhang','job':'Manager','report':'VP'},\
{'name':'Li','sal':5000,'report':'IT'}]
data_employee = pd.read_json(json.dumps(json_data))
data_employee_ri = data_employee.reindex(columns=['name',\
                                                  'job','sal','report'])


# %% pandas数据操纵
# pd.concat默认的参数类似于R语言的rbind, axis=1后类似于R语言中的cbind
pd.concat([data_employee_ri, data_employee_ri])
pd.concat([data_employee_ri, data_employee_ri], axis=1)

# merge
pd.merge(data_employee_ri, data_employee_ri, on='name')
pd.merge(data_employee_ri, data_employee_ri, on=['name', 'job'])

# %% 自定义函数映射
dataNumPy32 = np.asarray([('Japan','Tokyo',4000),('S.Korea','Seoul',1300),\
                          ('China','Beijing',9100)])
DF32 = pd.DataFrame(dataNumPy32,columns=['nation','capital','GDP'])

def GDP_Factorize(v):
    fv = np.float64(v)
    if fv>6000.0:
        return 'High'
    elif fv<2000.0:
        return 'Low'
    else:
        return 'Middle'

DF32['GDP_level'] = DF32['GDP'].map(GDP_Factorize)
DF32['nation'] = DF32['nation'].map(str.upper)

# %% 分组
# map函数：分组处理
# agg函数：输入多个值，返回一个值
iris_group = iris.groupby('class')

for levels, subdf in iris_group:
    print(levels)
    print(subdf)

iris.groupby('class').agg(\
            lambda x: ((x-x.mean())**3).sum()*len(x)/(len(x)-1)/(len(x)-2)/x.std()**3\
            if len(x)>2 else None)

pd.concat([iris, iris.groupby('class').transform('mean')], axis=1)

# %% 多级索引的透视操作
factor1 = np.random.randint(0,3,50)
factor2 = np.random.randint(0,2,50)
factor3 = np.random.randint(0,3,50)
values = np.random.randn(50)
hierindexDF = pd.DataFrame({'F1':factor1, 'F2':factor2, 'F3':factor3,\
                            'F4':values})
hierindexDF_gbsum = hierindexDF.groupby(['F1', 'F2', 'F3']).sum()
hierindexDF_gbsum
# unstack
# 无参数时，把最末index置换到column上
# 有数字参数时，把指定位置的index置换到column上
# 有列表参数时，依次把特定位置的index置换到column上
hierindexDF_gbsum.unstack(0)
hierindexDF_gbsum.unstack(1)
hierindexDF_gbsum.unstack([2,0])
hierindexDF_gbsum.unstack([2,0]).stack([1,2])
