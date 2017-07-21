# -*- coding: utf-8 -*-
# %%
import matplotlib.pyplot as plt
from random import choice
from random import randint
import pygal
import platform
import csv

# %%

win_path = r'D:\WorkSpace\CodeSpace\Code.Data\Python'
mac_path = r'D:\WorkSpace\CodeSpace\Code.Data\Python'
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

# %% 下载数据
filename = r'D:\WorkSpace\CodeSpace\Python\Python3\sitka_weather_07-2014.csv'

with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    
    highs = []
    for row in reader:
        high = int(row[1])
        highs.append(high)

for index, column_header in enumerate(header_row):
    print(index, column_header)
    
plt.plot(highs, c='red')
plt.title('Daily high tempeature, july 2014', fontsize=20)
plt.xlabel('', fontsize=16)
plt.ylabel('temperature', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
