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
import string
# %% 1、pandas导入、查看版本
print(pd.__version__)
print(pd.show_versions())
# %% 2、使用python list、python dict、numpy.ndarray创建pandas.Series
mylist = list(string.ascii_lowercase) # python list
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))

ser1 = pd.Series(mylist)
ser2 = pd.Series(myarr)
ser3 = pd.Series(mydict)

print(f'ser1 is \n{ser1}')
print(f'ser2 is \n{ser2}')
print(f'ser3 is \n{ser3}')
# %% 3、将pandas.Series转化为pandas.DataFrame
mylist = list(string.ascii_lowercase)
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))
ser = pd.Series(mydict)

# to_dframe()结合reset_index()使用
df = ser.to_frame().reset_index()
print(df.head())
# %% 4、将多个pandas.Series合并为一个pandas.DataFrame
ser1 = pd.Series(list(string.ascii_lowercase))
ser2 = pd.Series(np.arange(26))

# 解决办法1
df1 = pd.concat([ser1, ser2], axis=1)
# 解决办法2
df2 = pd.DataFrame({'col1':ser1, 'col2':ser2})
# %% 5、修改pandas.Series index名称
ser = pd.Series(list(string.ascii_uppercase))
ser.name = 'alphabets'
# %% 6、移除pandas.Series1中和pandas.Series2共同的部分
ser1 = pd.Series([1,2,3,4,5])
ser2 = pd.Series([4,5,6,7,8])
ser1[~ser1.isin(ser2)]
# %% 7、求pandas.Series1和pandas.Series2的交集、并集、差集
ser1 = pd.Series([1,2,3,4,5])
ser2 = pd.Series([4,5,6,7,8])
ser_u = pd.Series(np.union1d(ser1, ser2))
ser_i = pd.Series(np.intersect1d(ser1, ser2))
ser_s = ser_u[~ser_u.isin(ser_i)]

print(f'并集是\n{ser_u}')
print(f'交集是\n{ser_i}')
print(f'差集是\n{ser_s}')
# %% 8、求pandas.Series分位数（最小值、1/4分位数、中位数、3/4分位数、最大值）
state = np.random.RandomState(100)
ser = pd.Series(state.normal(10, 5, 25))
np.percentile(ser, [0, 25, 50, 75, 100])
print(np.percentile(ser, [0, 25, 50, 75, 100]))
# %% 9、求pandas.Series()频数
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
ser.value_counts()
print(ser.value_counts())
# %% 10、输出pandas.Series()中频数排第一二位的、其它的替换为other
np.random.RandomState(100)
ser = pd.Series(np.random.randint(1, 5, [12]))

print(f'top 2 value_counts: \n{ser.value_counts()}')
ser[~ser.isin(ser.value_counts().index[:2])] = 'other'
# %% 11、将pandas.Series()均分为10个区间、每个值使用区间名称标记
# %% 12、将pandas.Series()转换为指定shape的pandas.DataFrame
# %% 13、取出pandas.Series()中满足条件数据的位置index
# %% 14、取出pandas.Series()指定位置的数据
# %% 15、pandas.Series()水平、垂直合并
# %% 16、输出pandas.Series()子集的index号
# %% 17、求真实和预测pd.Series之间的均方误差损失函数（MSE，mean squared error)
# %% 18、pd.Series字符串型数据首字母大写转换
# %% 19、pd.Series字符串型数据字符长度计算
# %% 20、pd.Series中两两数之间差异
# %% 21、pd.Series中日期字符串转换为datetime格式
# %% 22、获取pd.Series日期字符串中时间对象
# %% 23、pd.Series日期字符串中修改为按指定日期输出
# %% 24、输出pd.Series中至少包含两个元音字符的数据
# %% 25、输出pd.Series中有效的email地址
# %% 26、pd.Series1按pd.Series2分组并求均值
# %% 27、计算两个pd.Series之间的欧式距离
# %% 28、求pd.Series局部峰值index
# %% 29、pd.Series字符串数据中使用最低频字符填充空格
# %% 30、创建时间序列数据，赋予随机值
# %% 31、缺省的时间序列值 不同方式填充
# %% 32、找出pd.Series中自相关性最大的数据
# %% 33、从一个csv 文件中每间隔50行取数据生成pandas.DataFrame
# %% 34、从一个csv 文件取数据生成pandas.DataFrame（新增加一分类列）
# %% 35、生成一个按规定步长平移的pandas.DataFrame
# %% 36、从一个csv 文件读取指定列生成pandas.DataFrame
# %% 37、输出DataFrame的行数、列数、数据类型、类型频数、Series转list
# %% 38、输出满足某个规则的DataFrame数据行和列号
# %% 39、修改DataFrame的列名称
# %% 40、DataFrame中是否有缺省值确认
# %% 41、DataFrame中缺省值统计
# %% 42、各自列均值填充DataFrame中各自列缺省值
# %% 43、各自列均值、中值填充DataFrame中各自列缺省值（使用apply）
# %% 44、从DataFrame选择子DataFrame
# %% 45、 改变DataFrame列顺序
# %% 46、大DataFrame修改默认显示的行和列数
# %% 47、DataFrame数据小数位数设置
# %% 48、 DataFrame数据小数转百分比显示
# %% 49、DataFrame数据每隔20行读取
# %% 50、创建DataFrame主键
# %% 51、获取DataFrame某一列中第n大的值索引
# %% 52、获取DataFrame某一列中第n大的值大于指定值得索引
# %% 53、获取DataFrame中行和大于100的行
# %% 54、 Series or DataFrame中使用分位数填充超限区域
# %% 55、去除指定值将DataFrame转换为最大方阵
# %% 56、DataFrame两行交换
# %% 57、DataFrame逆序输出
# %% 58、DataFrame转对角矩阵
# %% 59、DataFrame那一列含有最多行最大值
# %% 60、DataFrame创建新列：每行为行号（按欧几里得距离而来）
# %% 61、求DataFrame各列之间最大相关系数
# %% 62、DataFrame创建一列：包含每行中最小值与最大值比值
# %% 64、DataFrame每列按特定方式归一化
# %% 65、计算DataFrame每行与后一行的相关系数
# %% 66、DataFrame对角线元素替换为0
# %% 67、DataFrame按某列分组、提取某个分组
# %% 68、DataFrame按另外列分组、提取当前列中指定值（看下方例子，需求不好描述）
# %% 69、DataFrame分组（看下方例子，需求不好描述）
# %% 70、两个DataFrame使用类似SQL 中INNER JOIN拼接
# %% 72、取出DataFrame中两列值相等的行号
# %% 73、DataFrame中新建两列：滞后列和提前列（看下方例子，需求BT）
# %% 74、DataFrame中所有值出现频次统计
# %% 75、拆分DataFrame中某列文本为两列