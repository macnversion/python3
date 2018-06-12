# -*- coding: utf-8 -*

# %%
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

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