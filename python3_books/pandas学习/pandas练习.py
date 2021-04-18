#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:52:33 2021

@author: machuan
"""
# %% 加载需要的包
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib as plt

# %%

def stat(x):
    return pd.Series([x.count(), x.min(), x.max()],
                     index=['Count','min','max'])

# %%
stu_dic = {'Age':[14,13,13,14,14,12,12,15,13,12,11,14,12,15,16,12,15,11,15],
'Height':[69,56.5,65.3,62.8,63.5,57.3,59.8,62.5,62.5,59,51.3,64.3,56.3,66.5,
          72,64.8,67,57.5,66.5],
'Name':['Alfred','Alice','Barbara','Carol','Henry','James','Jane','Janet',
        'Jeffrey','John','Joyce','Judy','Louise','Marry','Philip','Robert',
        'Ronald','Thomas','Willam'],
'Sex':['M','F','F','F','M','M','F','F','M','M','F','F','F','F','M','M','M',
       'M','M'],
'Weight':[112.5,84,98,102.5,102.5,83,84.5,112.5,84,99.5,50.5,90,77,112,150,
          128,133,85,112]}
student = pd.DataFrame(stu_dic)

# %% 简单的数据查询
# 记得loc和iloc的区别在于loc是基于标签的值去索引，iloc基于标签的位置去索引
print(student.head())

# 所有12岁以上女生的信息
student[(student['Age']>=12)&(student['Sex']=='F')]
# 所有12岁以上女生的姓名、身高和体重
student[(student['Age']>=12)&(student['Sex']=='F')][['Name','Age','Weight']]

# %%
np.random.seed(1234)
d1 = pd.Series(2*np.random.normal(size = 100)+3)
d2 = np.random.f(2,4,size = 100)
d3 = np.random.randint(1,100,size = 100)

print('非空元素计算：', d1.count())
print('最大值：', d1.max())
print('min:', d1.min())
print('index of max:', d1.idxmax())
print('inde of min:', d1.idxmin())
print(stat(d1))


df = pd.DataFrame(np.array([d1,d2,d3]).T, columns=['x1','x2','x3'])
# %% pandas实现sql查询操作
dic = {'Name':['LiuShunxiang','Zhangshan'],
       'Sex':['M','F'],'Age':[27,23],
       'Height':[165.7,167.2],'Weight':[61,63]}
student2 = pd.DataFrame(dic)
student3 = pd.concat([student, student2])

print(pd.DataFrame(student2, columns=['Age','Weight', 'Height', 'Sex','Name', 'Score']))

# %%
a=[1,2,3,4]
b=[4,3,2,1]
plt.plot(a,b)