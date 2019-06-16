# -*- coding: utf-8 -*-

# %%
import os
import numpy as np
import scipy as sp
import pandas as pd
import scipy as sp
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %% 基本的几个包的应用
# scipy求解方程组
from scipy.optimize import fsolve
def f(x):
    x1 = x[0]
    x2 = x[1]
    return [2*x1 - x2**2 - 1, x1**2 - x2 -2]


result = fsolve(f, [1, 1])
print(result)

# %% 相关参数设定
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# %% 文件路径
sorted(os.listdir('./dataset/Python数据分析与挖掘实战'))

# %% catering数据异常值检测
catering_sales = pd.read_excel('./dataset/Python数据分析与挖掘实战/catering_sale.xls', index_col=u'日期')
catering_sales.describe()
# %%
plt.figure()
p = catering_sales.boxplot()
#x = p['fliers'][0].get_xdata()  # 不确定语法的正确性
#y = p['fliers'][0].get_ydata()
#y.sort()
plt.show()
# %%
catering_sales_clean = catering_sales[(catering_sales['销量']>400)&(catering_sales['销量']<5000)]
statistics = catering_sales_clean.describe()  # 生成的statistics类型是data frame
statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']
statistics.loc['var'] = statistics.loc['std']/statistics.loc['mean']
statistics.loc['dis'] = statistics.loc['75%'] - statistics.loc['25%']
print(statistics)

# %% 帕累托分析
dish_profit = pd.read_excel('./dataset/Python数据分析与挖掘实战/catering_dish_profit.xls',
                            index_col=u'菜品名')
# %%
dish_profit_copy = dish_profit['盈利']
dish_profit_copy.sort_values(ascending=False)

plt.figure()
dish_profit_copy.plot(kind='bar')
plt.ylabel('盈利（元）')
p = dish_profit_copy.cumsum()/dish_profit_copy.sum()
p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
plt.ylabel('盈利（比例）')
plt.show()
# %% 相关性分析
catering_sales = pd.read_excel()
# %% 连续属性离散化
discretization_data = pd.read_excel('./dataset/Python数据分析与挖掘实战/discretization_data.xls')
data = discretization_data[u'肝气郁结证型系数'].copy()
k = 4

d1 = pd.cut(data, k, labels=range(4))  # 等宽离散化

w = [1.0*i/k for i in range(k+1)]
w = data.describe(percentiles = w)[4:4+k+1]
w[0] = w[0]*(1-1e-10)
d2 = pd.cut(data, w, labels=range(k))  # 等频率离散化


kmodel = KMeans(n_clusters = k, n_jobs = 4)  # 建立模型，n_jobs是并行数
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

# %% 逻辑回归
bankloan = pd.read_excel('./dataset/Python数据分析与挖掘实战/bankloan.xls')
xx = bankloan.iloc[:, :8]
yy = bankloan.iloc[:, 8]
x = xx.as_matrix()
y = yy.as_matrix()


from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

rlr = RLR()
rlr.fit(x, y)
rlr.get_support()
print(u'通过随机逻辑回归模型筛选特征结束 。')
print(u'有效特征为：%s' % ','.join(xx.columns[rlr.get_support()]))

x = xx[xx.columns[rlr.get_support()]].as_matrix()
lr = LR()
lr.fit(x, y)
print(u'逻辑回归模型训练结束')
print(u'模型平均正确率为:%s' % lr.score(x, y))

# %% 决策树
sales_data = pd.read_excel('./dataset/Python数据分析与挖掘实战/sales_data.xls')
sales_data[sales_data == u'好'] = 1
sales_data[sales_data == u'是'] = 1
sales_data[sales_data == u'高'] = 1
sales_data[sales_data != 1] = -1

x = sales_data.iloc[:, :3].as_matrix().astype(int)
y = sales_data.iloc[:, :3].as_matrix().astype(int)

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy')
dtc.fit(x,y)

# %% 电力窃漏用户自动识别
model_data = pd.read_excel('./dataset/Python数据分析与挖掘实战/model.xls')