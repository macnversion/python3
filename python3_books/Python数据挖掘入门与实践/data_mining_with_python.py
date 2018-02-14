# -*- coding: utf-8 -*-
# python数据挖掘入门与实践
# %%
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from collections import defaultdict
# %% 显示全部的数据集的名称
sorted(os.listdir('./dataset/python数据挖掘入门与实践'))
# %%
X = np.loadtxt('./dataset/python数据挖掘入门与实践/affinity_dataset.txt')
features = ['面包', '牛奶', '奶酪', '苹果', '香蕉']

# 以购买苹果为例，计算购买苹果的cases
num_apple_purchases = 0
for sample in X:
    if sample[3] == 1:
        num_apple_purchases += 1
print('{0} people bought apples'.format(num_apple_purchases))

# 既买苹果又买了香蕉
rule_valid = 0
rule_invalid = 0
for sample in X:
    if sample[3] == 1:
        if sample[4] == 1:
            rule_valid += 1
        else:
            rule_invalid += 1
print('{0} cases of the rule being valid were discovered'.format(rule_valid))
print('{0} cases of the rule bing invalid were discovered'.format(rule_invalid))

# 计算支持度和置信度
support = rule_valid
confidence = rule_valid/num_apple_purchases

valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)

for sample in X:
    for premise in range(4):
        if sample[premise] == 0:
            continue
