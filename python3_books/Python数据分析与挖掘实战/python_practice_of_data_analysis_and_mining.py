# -*- coding: utf-8 -*-

# %%
import os
import pandas as pd
import scipy as sp
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# %% 相关参数设定
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# %% 文件路径
sorted(os.listdir('./dataset/Python数据分析与挖掘实战'))

# %% catering数据异常值检测
catering_sales = pd.read_excel('./dataset/Python数据分析与挖掘实战/catering_sale.xls', index_col=u'日期')

plt.figure()
p = catering_sales.boxplot()
#x = p['fliers'][0].get_xdata()
#y = p['fliers'][0].get_ydata()
#y.sort()

catering_sales = catering_sales[(catering_sales[u'销量']>400)&(catering_sales[u'销量']<5000)]
statistics = catering_sales.describe()
statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']

# %% 帕累托分析
dish_profit = pd.read_excel('./dataset/Python数据分析与挖掘实战/catering_dish_profit.xls',
                            index_col = u'菜品名')
dish_profit = dish_profit[u'盈利'].copy()
dish_profit.sort_values(ascending = False)
plt.figure()
dish_profit.plot(kind='bar')
plt.ylabel(u'盈利（元）')
p = 1.0*dish_profit.cumsum()/dish_profit.sum()
p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
plt.ylabel(u'盈利（占比）')

# %% 连续属性离散化
discretization_data = pd.read_excel('./dataset/Python数据分析与挖掘实战/discretization_data.xls')
data = discretization_data[u'肝气郁结证型系数'].copy()
k = 4

d1 = pd.cut(data, k, labels=range(4)) # 等宽离散化

w = [1.0*i/k for i in range(k+1)]
w = data.describe(percentiles = w)[4:4+k+1]
w[0] = w[0]*(1-1e-10)
d2 = pd.cut(data, w, labels=range(k)) # 等频率离散化


kmodel = KMeans(n_clusters = k, n_jobs = 4) # 建立模型，n_jobs是并行数
kmodel.fit(data.values.reshape((len(data), 1)))
c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
w = pd.rolling_mean(c, 2).iloc[1:]
w = [0] + list(w[0]) + data.max()
d3 = pd.cut(data, w, labels=range(k))

# %% 相关性分析
catering_sale = pd.read_excel('./dataset/Python数据分析与挖掘实战/catering_sale_all.xls', index_col=u'日期')
catering_sale.corr()
catering_sale.corr()[u'百合酱蒸凤爪']
catering_sale[u'百合酱蒸凤爪'].corr(catering_sale[u'翡翠蒸香茜饺'])

# %% lagrange插值法
catering_sale = pd.read_excel('./dataset/Python数据分析与挖掘实战/catering_sale.xls')
catering_sale[u'销量'][(catering_sale[u'销量']<400)|(catering_sale[u'销量']>5000)] = None

def ployinterp_column(s, n, k=5):
    y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)

for i in catering_sale.columns:
    for j in range(len(catering_sale)):
        if (catering_sale[i].isnull())[j]:
            catering_sale[i][j] = ployinterp_column(catering_sale[i], j)

# %%
