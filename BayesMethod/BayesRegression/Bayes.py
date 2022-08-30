from GaussianProcess.GPR import getmeanandsigma
from BayesRegression.func import func
import numpy as np

UPPER = 10
LOWER = -1
from random import random
INITSIZE = 4
X = []
Y = []
BATCHSAMPLENUMBER = 16

EPOCH = 32 * 64
def _getrandom():
    return (UPPER - LOWER) * random() + LOWER
def _getinitrandom():
    return (100 - (-10)) * random() - 1

def _init():
    for _ in range(INITSIZE):
        xvalue = _getinitrandom()
        X.append(xvalue)
        Y.append(func(xvalue))

def acquire(mu,sigmamatrix:np.matrix,sigma = 1):
    # print(mu.shape)
    # print(np.diagonal(sigmamatrix).shape)
    # print(mu.squeeze(0))
    # exit()
    mu = np.reshape(mu,mu.shape[1])
    evaluation = mu + np.diagonal(sigmamatrix) * sigma
    argmin = np.argmax(evaluation)
    return argmin
def train():
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./log/Bayesoptim/Max')
    sample = []
    _init()
    from tqdm import tqdm
    for epoch in tqdm(range(EPOCH)):
        writer.add_scalar("value",max(Y),epoch)
        for _ in range(BATCHSAMPLENUMBER):
            sample.append(_getrandom())
        mu,sigma = getmeanandsigma(np.array(X),np.array(Y),np.array(sample))
        mu = mu.squeeze()
        index = acquire(mu,sigma)
        X.append(sample[index])
        Y.append(func(sample[index]))

    pass

if __name__ == "__main__":
    train()