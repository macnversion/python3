# -*- coding: utf-8 -*-

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
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

# %% 相关性分析
catering_sale = pd.read_excel('./dataset/Python数据分析与挖掘实战/catering_sale_all.xls', index_col=u'日期')
catering_sale.corr()
catering_sale.corr()[u'百合酱蒸凤爪']
catering_sale[u'百合酱蒸凤爪'].corr(catering_sale[u'翡翠蒸香茜饺'])