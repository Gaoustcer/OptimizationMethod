import torch
from random import random
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/Gradient_descent')
wlist = []
variablelist = []
N = 2
for _ in range(N):
    variablelist.append(torch.tensor(0.0,requires_grad=True,device='cuda:0'))
    wlist.append(random())
# result = torch.tensor(0.0,requires_grad=True)
results = []
EPOCH = 256
lr = 0.01
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
itertimes = []
itervalues = []
from tqdm import tqdm
for epoch in tqdm(range(EPOCH)):
    results = []
    for i in range(N-1):
        # result = torch.add(result,torch.add(torch.mul(wlist[i],torch.pow(torch.add()))))
        results.append(wlist[i]*(variablelist[i+1]-variablelist[i]**2)**2+(variablelist[i]-1)**2)
    # result = torch.stack(results)
    # print(result)
    loss = torch.sum(torch.stack(results))
    for tensor in variablelist:
        if tensor.grad is not None: 
            tensor.grad.data.zero_()
    loss.backward()
    # print("loss is",loss)
    for tensor in variablelist:
        # tensor = (tensor - lr * tensor.grad).detach().requires_grad_().cuda()
        tensor.data.add_(-lr * tensor.grad)
        # tensor.grad.zero_()
    itertimes.append(epoch)
    itervalues.append(float(loss))
    writer.add_scalar('Rosenvalue',float(loss),epoch)
    # itervalues.append(float(variablelist[0]))
plt.scatter(x=itertimes,y=itervalues)
plt.savefig("./gradient_descent_Rosen{}.png".format(str(N)))
import numpy as np

np.save('weight.npy',np.array(wlist))
# print("loss is",loss)
# print("loss device is",loss.device)
# loss.backward()
# print(variablelist[0].grad)
