from random import random
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log/zeroone')
count = {}

import matplotlib.pyplot as plt

N = 1024
M = 4096
for i in range(N+1):
    count[i] = 0
from tqdm import tqdm
for _ in range(M):
    positivetime = 0
    for _ in range(N):
        if random() < 0.5:
            positivetime += 1
    count[positivetime] += 1
plt.scatter(x = count.keys(),y = count.values())
plt.savefig("./onezerodis.jpg")
plt.show()
for key in range(N):
    writer.add_scalar('distribution',count[key],key)
    
        