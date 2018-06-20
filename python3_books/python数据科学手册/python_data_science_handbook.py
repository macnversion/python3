# -*- coding: utf-8 -*

# %%
import os
import platform
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn

# %% 设置工作路径
win_path = r'D:/WorkSpace/CodeSpace/Python/Python3'
if 'Windows' in platform.platform():
    data_path = win_path

# %% 计算代码的执行时间
# %timeit L = [n ** 2 for n in range(100)]

# %% numpy
# numpy中的可以直接应用的数组
np.zeros(10 ,dtype=int) # 创建长度为10的零数组
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
#big_array = np.random.randint(1, 100, size=1000000)
#timeit compute_reciprocals(big_array) # run this in console

# 通用函数的高级特性
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y) # out参数指定输出的位置

# 聚合
M = np.random.random((3,4))
M.min(axis=0)

# 广播
X = np.random.random((10, 3))
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# 布尔掩码
rainfall = pd.read_csv('./dataset/python数据科学手册/Seattle2014.csv')['PRCP'].values
inches = rainfall/254
seaborn.set()
plt.hist(inches, 40)
rainy = (inches > 0)

# 花式索引
rand = np.random.RandomState(42)
x = np.random.randint(100, size=10)
ind = np.array([[3, 7],
                [4, 5]])
    
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
    return x

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