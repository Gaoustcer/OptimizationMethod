import matplotlib.pyplot as plt

from torch.distributions import Normal
import torch
normalfunc = Normal(0,1)
x_list = []
y_list = []
N = 128 ** 2
x_list = normalfunc.sample_n(N)
y_list = normalfunc.sample_n(N)
plt.scatter(x_list,y_list,c='r',s=0.1)
plt.savefig("2dimensionGaussian.png")
normalfunc = Normal(torch.tensor([0,0]).to(torch.float32),torch.tensor([1,1]).to(torch.float32))
