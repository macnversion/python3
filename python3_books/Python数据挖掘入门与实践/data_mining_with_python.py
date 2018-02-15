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
n_sample, n_features = X.shape
features = ['面包', '牛奶', '奶酪', '苹果', '香蕉'] # 数据集的列对应的水果

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
# 计算所有可能的组合
for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0:
            continue
        num_occurances[premise] += 1
        for conclusion in range(n_features):
            if sample[premise] == conclusion:
                continue
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise, conclusion)] += 1
support = valid_rules
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    confidence[(premise, conclusion)] = valid_rules[(premise, conclusion)]/num_occurances[premise]

for primise, conclusion in confidence:
    primise_name = features[primise]
    conclusion_name = features[conclusion]
    print('Rule: If a persion buy {0} they will also buy {1}'.format(primise_name, conclusion_name))
    print('- Confidence:{0:.3f}'.format(confidence[(primise, conclusion)]))
    print('- Support:{0}'.format(support[(primise, conclusion)]))
    print('\n')

def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print('Rule: If a person buy {0} they will also buy {1}'.format(premise_name, conclusion_name))
    print('- Support: {0}'.format(support[(premise, conclusion)]))
    print('- Confidence: {0:.3f}'.format(confidence[(premise, conclusion)]))
    print('\n')

# 排序找出最佳的规则
from operator import itemgetter
sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)
for index in range(n_features):
    print('Rules #{0}'.format(index+1))
    (premise, conclusion) = sorted_support()
