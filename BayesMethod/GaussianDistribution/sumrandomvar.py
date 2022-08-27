from random import random
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
N = 10
values = []
NumberSample = 128 ** 2
for _ in range(NumberSample):
    average = 0
    for _ in range(N):
        average += random()
    values.append(average/N)
hist(values,bins=1024)
plt.savefig('averageofvar.png')