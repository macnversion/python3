# -*- coding: utf-8 -*-
from collections import defaultdict
from collections import Counter
import string


# %% different types of counter
s = """
Write a program that prints the numbers from 1 to 100.
But for multiples of three print 'Fizz' instead of the number and f
or the multiples of five print 'Buzz'. For numbers which are
multiples of both three and five print 'FizzBuzz'
"""

print(s.lower().count('a'))

counter1 = {}
for _ in s.lower():
    counter1[_] = counter1.get(_, 0) + 1
print(counter1['a'])


counter2 = defaultdict(int)
for _ in s.lower():
    counter2[_] += 1
print(counter2['a'])


counter3 = Counter(s.lower())
print(counter3['a'])

# %% Recursion
def fact(n):
    if n==0:
        return 1
    else:
        return n*fact(n-1)

# Fibonacci 
def fib1(n):
    if n==0 or n==1:
        return 1
    else:
        return fib1(n-1)+fib1(n-2)
# %% Generators
def count_down(n):
    for i in range(n,0,-1):
        yield i
counter = count_down(10)
print(counter.next())