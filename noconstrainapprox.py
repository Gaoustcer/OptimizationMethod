import numpy as np

weight = np.load('weight.npy',allow_pickle=True)
w = weight[0]
def function(x1,x2):
    return (1-x1)**2 + w*(x2-x1**2)**2

EPOCH = 256
def update(x1,x2):
    def getgradient():
        return -np.array([4*w*x1**3-4*w*x1*x2+2*x1-2,2*w*(x2 - x1**2)])
    def gethessianmatrix():
        return np.array(
            [
                [12 * w * x1**2-4*w*x2 +2,-4*w*x1],
                [-4*w*x1,2*w]
            ]
        )
    matrix = gethessianmatrix()
    gradient = getgradient()
    reverse = np.linalg.inv(matrix)
    delta = reverse.dot(gradient)
    # print()
    return x1 + delta[0], x2 + delta[1]
x1 , x2 = 0,0
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./log/noconstrainapprox")
from tqdm import tqdm
for epoch in tqdm(range(EPOCH)):
    # x1 = 0
    # x2 = 0
    result = function(x1,x2)
    print("result is {}".format(str(epoch)),result)
    print("x1 x2 is",x1,x2)
    writer.add_scalar("Rosenvalue",result,epoch)
    x1,x2 = update(x1,x2)
