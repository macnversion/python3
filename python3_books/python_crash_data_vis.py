# -*- coding: utf-8 -*-
# python crash course and 利用python进行数据分析
# %%
import matplotlib.pyplot as plt
from random import choice
from random import randint
#import pygal
import platform
import csv
from datetime import datetime
import json
import requests
from collections import defaultdict
from collections import Counter
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import os
import re
#from pandas_datareader import data, wb

# %%

win_path = r'D:/WorkSpace/CodeSpace/Code.Data/Python/'
mac_path = r'/Users/machuan/CodeSpace/Code.Data/python/'
data_path = win_path if 'Windows' in platform.platform() else mac_path

# %% 折线图
input_value = [i for i in range(1, 6)]
squares = [i**2 for i in input_value]
plt.plot(input_value, squares, linewidth=2)
plt.title('square of numbers', fontsize=16)
plt.xlabel('value', fontsize=16)
plt.ylabel('squarof value', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.show()

# %% 散点图
x_values = [i for i in range(1, 10)]
y_values = [i**2 for i in x_values]
z_values = [2*i for i in x_values]
plt.scatter(x_values, y_values, s=20, edgecolor='red', c='yellow')
plt.axis([0, 15, 0, 300])
plt.title('scatter plot', fontsize=20)
plt.xlabel('x_value', fontsize=16)
plt.ylabel('y_value', fontsize=16)
plt.tick_params(axis='both', labelsize=10)

plt.scatter(x_values, z_values, s=20)

# %% 随机漫步
class RandomWalk():
    '''定义一个随机漫步的类'''
    def __init__(self, num_points=5000):
        '''初始化随机漫步的属性'''
        self.num_points = num_points
        
        # 所有的随机漫步都起始于[0, 0]
        self.x_values = [0]
        self.y_values = [0]

    
    def fill_walk(self):
        # 不断漫步，直到达到指定长度
        while(len(self.x_values)<self.num_points):
            x_direction = choice([-1, 1])
            x_distance = choice(range(0, 5))
            x_step = x_direction * x_distance
            
            y_direction = choice([-1, 1])
            y_distance = choice(range(0, 5))
            y_step = y_direction * y_distance
            
            if x_step==0 and y_step==0:
                continue
            
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step
            
            self.x_values.append(next_x)
            self.y_values.append(next_y)


while True:
    rw = RandomWalk()
    rw.fill_walk()
    point_numbers = list(range(rw.num_points))
    point_numbers_process = list(range(rw.num_points-2))
    
    # 突出起点和终点
    plt.scatter(rw.x_values[0], rw.y_values[0], c='red', \
                edgecolor='none', s=100)
    plt.scatter(rw.x_values[-1], rw.y_values[-1], c='yellow', 
                edgecolor='none', s=100)
    plt.scatter(rw.x_values[1:-1], rw.y_values[1:-1], s=10,\
                c=point_numbers_process, cmap=plt.cm.Blues,\
                edgecolor='none')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.show()
    
    keep_running = input('Make another walk?(Y/N):')
    if keep_running == 'n':
        break

# %% Pygal模拟掷骰子
class Die():
    '''表示一个掷骰子的类'''
    def __init__(self, num_sides=6):
        self.num_sides=num_sides
        
    def roll(self):
        return(randint(1, self.num_sides))


die = Die()
results = []
for roll_num in range(1, 1000):
    result = die.roll()
    results.append(result)

print(results, '\n')

frequencise = []
for value in range(1, die.num_sides+1):
    frequency = results.count(value)
    frequencise.append(frequency)

print(frequencise)

hist = pygal.Bar()
hist.title = 'results of rolling D6 1000 times'
hist.x_lables =  ['1', '2', '3', '4', '5', '6']
hist.x_title = 'values'
hist.y_title = 'frequency of values'
hist.add('D6', frequencise)
hist.render_to_file('die_visual.svg')

# 抛两个骰子
die1 = Die()
die2 = Die()
results = []

for roll_num in range(1000):
    result = die1.roll() + die2.roll()
    results.append(result)

frequencies = []
max_result = die1.num_sides + die2.num_sides
for value in range(2, max_result):
    frequency = results.count(value)
    frequencies.append(frequency)

hist = pygal.Bar()
hist.title = 'results of rolling two D6 1000 times'
hist.x_labels = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
hist.x_title = 'value'
hist.y_value = 'frequency of value'

hist.add('D6+D6', frequencies)
hist.render_to_file('die_visual.svg')

# %% 下载数据-csv文件的处理
filename = 'sitka_weather_07-2014.csv'

with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    
    dates = []
    highs = []
    for row in reader:
        current_date = datetime.strptime(row[0], '%Y-%m-%d')
        dates.append(current_date)
        
        high = int(row[1])
        highs.append(high)

for index, column_header in enumerate(header_row):
    print(index, column_header)

fig1 = plt.figure(dpi=128)    
plt.plot(dates, highs, c='red')
plt.title('Daily high tempeature, july 2014', fontsize=12)
plt.xlabel('dates', fontsize=10)
fig1.autofmt_xdate()
plt.ylabel('temperature', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=8)


filename = 'sitka_weather_2014.csv'

with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    
    dates = []
    highs = []
    lows = []
    for row in reader:
        current_date = datetime.strptime(row[0], '%Y-%m-%d')
        dates.append(current_date)
        
        high = int(row[1])
        highs.append(high)
        
        low = int(row[3])
        lows.append(low)

fig2 = plt.figure(dpi=128)
plt.plot(dates, highs, c='red')
plt.plot(dates, lows, c='blue')
plt.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)
plt.title('Daily high and low temperature 2014', fontsize=12)
plt.xlabel('time', fontsize=10)
fig2.autofmt_xdate()
plt.ylabel('temp', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=8)

# %% json数据处理
filename = 'population_data.json'
with open(filename) as f:
    popdata = json.load(f)

for pop_dict in popdata:
    if pop_dict['Year']=='2010':
        countryname = pop_dict['Country Name']
        population = int(float(pop_dict['Value']))
        print(countryname + ':' + str(population))

# %% 使用API
url = 'https://api.github.com/search/repositories?' + \
'q=language:python&sort=stars'

r = requests.get(url)
print('Status Code:', r.status_code)

response_dict = r.json()
print(response_dict.keys(), '\n')

repo_dicts = response_dict['items']
print('Repositories returned:', len(repo_dicts))

repo_dict = repo_dicts[0]
for key in sorted(repo_dict.keys()):
    print(key)


# %%  利用python进行数据分析-usa.gov数据
path = data_path + '利用Python进行数据分析/ch02/' + \
'usagov_bitly_data2012-03-16-1331923249.txt'

records = [json.loads(line) for line in open(path)]
time_zones = [record['tz'] for record in records if 'tz' in record]


def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts


def top_counts(cdict, n=10):
    value_key_pairs = [(counts, tz) for tz, counts in cdict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]


counts = get_counts(time_zones)
top_counts(counts)

counts2 = Counter(time_zones)
counts2.most_common(10)

frame = DataFrame(records)
tz_counts = frame['tz'].value_counts()
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz==''] = 'Unknown'
tz_counts = clean_tz.value_counts()


result = Series([x.split()[0] for x in frame.a.dropna()])
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),
                            'Windows', 'Not Windows')
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
indexer = agg_counts.sum(1).argsort()

# %% 利用python进行数据分析-movielens 1M数据
movie_path = data_path + '利用Python进行数据分析/ch02/movielens' + \
'/movies.dat'
ratings_path = data_path + '利用Python进行数据分析/ch02/movielens' + \
'/ratings.dat'
users_path = data_path + '利用Python进行数据分析/ch02/movielens' + \
'/users.dat'

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
mnames = ['movie_id', 'title', 'genres']

movies = pd.read_table(movie_path, sep='::', header=None, names=mnames,
                      engine='python')
ratings = pd.read_table(ratings_path, sep='::', header=None, names=rnames,
                      engine='python')
users = pd.read_table(users_path, sep='::', header=None, names=unames,
                      engine='python')

data = pd.merge(pd.merge(ratings, movies), users)
mean_rating = data.pivot_table('rating', index='title', columns='gender',
                               aggfunc='mean')
rating_by_title = data.groupby('title').size()
active_titles = rating_by_title[rating_by_title>250]
mean_rating = mean_rating.iloc[active_titles]

top_female_rating = mean_rating.sort_values(by='F', ascending=False)

mean_rating['diff'] = mean_rating['M'] - mean_rating['F']
sort_by_diff = mean_rating.sort_values(by='diff')

rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.loc[active_titles]


# %% 利用python进行数据分析-全美婴儿姓名
yob1880_path = data_path + '利用Python进行数据分析/ch02/names' + \
'/yob1880.txt'
names1880 = pd.read_csv(yob1880_path, names=['name', 'sex', 'births'])
names1880.groupby('sex').births.sum()

years = range(1980, 2011)
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = data_path + '利用Python进行数据分析/ch02/names' + '/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)

names = pd.concat(pieces, ignore_index=True)

total_births = names.pivot_table('births', index='year', columns='sex',
                                 aggfunc='sum')
total_births.plot(title='Total births by year and sex')


def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births/births.sum()
    return group

names = names.groupby(['year', 'sex']).apply(add_prop)
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)


def get_top1000(group):
    return group.sort_values(by='births', ascending=True)[:1000]

top1000 = names.groupby(['year', 'sex']).apply(get_top1000)
boys = top1000[top1000.sex=='M']
girls = top1000[top1000.sex=='F']
total_births = top1000.pivot_table('births', index='year', columns='name',
                                   aggfunc=sum)

subset = total_births[['Aaban', 'Aabid', 'Aadhav', 'Aabriella']]

# %% 利用python进行数据分析-numpy and pandas
genre_iters = (set(x.split('|')) for x in movies.genres)
genres = sorted(set.union(*genre_iters))

dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)

for i, gen in enumerate(movies.genres):
    dummies.loc[i, gen.split('|')] = 1

movies_windic = movies.join(dummies.add_prefix('Genre_'))


# 正则表达式
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""

pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)

pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)

# %% usa食品数据库
json_file_path = data_path + \
    '利用Python进行数据分析/ch07/' + \
    'foods-2011-10-03.json'
db = json.load(open(json_file_path))

nutrients = db[0]['nutrients']
info_keys = ['description', 'group', 'id', 'manufacturer']
info = DataFrame(db, columns=info_keys)

nutrients = []
for rec in db:
    funts = DataFrame(rec['nutrients'])
    funts['id'] = rec['id']
    nutrients.append(funts)

nutrients = pd.concat(nutrients, ignore_index=True)
nutrients = nutrients.drop_duplicates()

col_mapping = {'description':'food',
               'group':'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
col_mapping = {'description':'nutrient',
               'group':'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
ndata = pd.merge(info, nutrients, on='id', how='outer')

result = ndata.groupby(['food', 'fgroup'])['value'].quantile(0.5)


# %% matplotlib 基础
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
_ = ax1.hist(np.random.randn(100), bins=20, color='r', alpha=0.5)
ax2 = fig.add_subplot(2,2,2)
ax2.scatter(np.arange(30), np.arange(30) + 3*np.random.randn(30))
ax3 = fig.add_subplot(2,2,3)
plt.plot(np.random.randn(50).cumsum(), 'k--')

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                            rotation=30, fontsize='small')
ax.set_title('a matplotlib test result')
ax.set_xlabel('Stage')

# %% 数据聚合与分组计算
def peak_to_peak(arr):
    return(arr.max() - arr.min())


os.chdir(data_path + r'利用Python进行数据分析/ch08')
tips = pd.read_csv('tips.csv')
tips['tip_pct'] = tips['tip']/tips['total_bill']

grouped = tips.groupby(['sex', 'smoker'])
grouped_pct = grouped['tip_pct']

functions = ['count', 'mean', 'max']
result = grouped['total_bill', 'tip_pct'].agg(functions)

def top(df, n=5, column='tip_pct'):
    return(df.sort_values(by=column)[-n:])


tips.groupby('smoker').apply(top)
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')

# 分位数和桶分析
frame = DataFrame({'data1':np.random.randn(100),
                   'data2':np.random.randn(100)})


# 随机采样和排列
# 红桃：hearts， 黑桃：spades， 梅花：clubs， 方片：diamonds
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10]*3)*4
base_names = ['A'] + list(range(2, 11)) + ['J', 'Q', 'K']
cards = []

for suit in suits:
    cards.extend(str(num) + suit for num in base_names)

deck = Series(card_val, index=cards)


def draw(deck, n=5):
    return(deck.take(np.random.permutation(len(deck)))[:n])


get_suit = lambda card: card[-1]
deck.groupby(get_suit).apply(draw)

# 示例加权平均数和相关系数
os.chdir(data_path + r'利用Python进行数据分析/ch09')
close_px = pd.read_csv('stock_px.csv', parse_dates=True, index_col=0)
rets = close_px.pct_change().dropna()
spx_corr = lambda x: x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x: x.year)
by_year.apply(spx_corr)


# %% 透视表和交叉表
# 透视表和交叉表

tips.pivot_table(['total_bill', 'size'],
                 index=['sex', 'day'],
                 columns='smoker')

# 示例：联邦选举委员会数据库
fec = pd.read_csv('P00000001-ALL.csv')
unique_cand = fec.cand_nm.unique()
#fec['party'] = fec.cand_nm.map(parties)


# %% 时间序列

now = datetime.now()
from dateutil.paeser import parse

longer_ts = Series(np.random.randn(1000),
                   index=pd.date_range('1/1/2007', periods=1000))
