# -*- coding: utf-8 -*-

# %%
import numpy as np
import matplotlib.pyplot as plt 
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
plt.imshow(z)