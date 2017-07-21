# -*- coding: utf-8 -*-
# %%
import matplotlib.pyplot as plt
from random import choice

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
    
    # 突出起点和终点
    plt.scatter(rw.x_values[0], rw.y_values[0], edgecolor='none', s=100)
    plt.scatter(rw.x_values[-1], rw.y_values[-1], edgecolor='none', s=100)
    plt.scatter(rw.x_values, rw.y_values, s=10,\
                c=point_numbers, cmap=plt.cm.Blues, edgecolor='none')
    plt.show()
    
    keep_running = input('Make another walk?(Y/N):')
    if keep_running == 'n':
        break
