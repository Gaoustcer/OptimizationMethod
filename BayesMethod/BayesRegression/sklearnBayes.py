from sklearn import gaussian_process
surrogate = gaussian_process.GaussianProcessRegressor()
from BayesRegression.func import func
from random import random
import numpy as np

INIT_SIZE = 4
SEARCHSIZE = 16
X = []
Y = []
def _getrandom():
    START = -1
    END = 10
    return (END - START) * random() + START

def _init():
    for _ in range(INIT_SIZE):
        xvalue = _getrandom()
        X.append(xvalue)
        Y.append(func(xvalue))
def acquire(mu,sigma):
    return np.argmax(mu + sigma)


if __name__ == "__main__":
    _init()
    EPOCH = 128
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./log/baselinebayes')
    from tqdm import tqdm
    for epoch in tqdm(range(EPOCH)):
        sample = 11 * np.random.random((SEARCHSIZE,1)) -1
        
        surrogate.fit(np.array([X]).T,np.array([Y]).T)
        mu,sigma = surrogate.predict(sample,return_std=True)
        index = acquire(mu,sigma)
        X.append(sample[index])
        Y.append(func(sample[index]))
        writer.add_scalar('value',max(Y),epoch)



    