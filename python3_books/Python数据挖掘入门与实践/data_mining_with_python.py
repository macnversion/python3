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
# %% 相关性分析示例
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
    (premise, conclusion) = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)

# %% 分类问题的简单示例
from sklearn.datasets import load_iris
from collections import defaultdict
from operator import itemgetter
dataset = load_iris()
X = dataset.data
y = dataset.target
print(dataset.DESCR, '\n')

n_sample, n_features = X.shape
attribute_means = X.mean(axis=0)
assert attribute_means.shape == (n_features, )
X_d = np.array(X >= attribute_means, dtype='int')

# 将数据拆分为训练集和测试集
from sklearn.cross_validation import train_test_split
random_state = 14 # 随机种子
X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=random_state)
print('There are {} train samples'.format(y_train.shape))
print('There are {} test samples'.format(y_test.shape))

def train_feature_value(X, y_true, feature, value):
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_true):
        if sample[feature] == value:
            class_counts[y] += 1
    # 获取最好的因素
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    n_samples = X.shape[1]
    error = sum([class_count for class_value, class_count in class_counts.items() \
                 if class_value != most_frequent_class])
    return most_frequent_class, error


def train(X, y_true, feature):
    """
    parameter
    X: array[n_samples, n_features]
    y_true: array[n_samples]
    features: int
    ------------
    return
    predictors: dictionary if tuples: (values, prediction)
    error: float
    """
    n_samples, n_features = X.shape
    assert 0 <= feature < n_features
    values = set(X[:, feature])
    predictors = dict()
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error

# compute all of the predictors
all_predictors = {variable: train(X_train, y_train, variable) for variable in range(X_train.shape[1])}
# to be countinue

# %% 用scikit-learn估计器分类
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt

X = np.zeros((351, 34), dtype='float')
y = np.zeros((351, ), dtype='bool')

with open('./dataset/python数据挖掘入门与实践/ionosphere.data', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        y[i] = row[-1] == 'g'

random_state = 14
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
print('There are {} samples in training dataset'.format(X_train.shape[0]))
print('There are {} sample in test dataset'.format(X_test.shape[0]))
print('Each sample has {} features'.format(X_train.shape[1]))

estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)
y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted)*100 
print('The accuracy is {0:.1f}%'.format(accuracy))

scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print('The average accuracy is {0:.1f}%'.format(average_accuracy))

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21))
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)
    
plt.plot(parameter_values, avg_scores, '-o')

# %% 用决策树预测球队获胜
dataset = pd.read_csv('./dataset/python数据挖掘入门与实践/leagues_NBA_2014_games_games.csv',
                      parse_dates=['Date'])
dataset.columns = ['Date', 'ScoreType', 'VisitorTeam', 'VisitorPts', 'HomeTeam', 'HomePts', 'OT?', 'Notes']

# %% test 代码测试
pd.read_csv('./dataset/python数据挖掘入门与实践/ionosphere.data', header=None).as_matrix()