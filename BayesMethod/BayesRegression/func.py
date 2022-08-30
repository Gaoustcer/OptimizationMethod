from matplotlib import pyplot as plt
from math import sin
from random import random
from math import exp
def func(x):
    return exp(-(x-2)**2) + exp(-(x-6)**2/10)+1/(x**2+1)
# def func(x):
#     return x**2 * (sin(x) ** 4) + random()/1024
import numpy as np

# x_list = np.arange(-1,10,0.001)
# y_list = []
# for x in (x_list):
#     y_list.append(func(x))
# plt.scatter(x_list,y_list)
# plt.savefig("funcfig.png")