from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import numpy as np

weight = np.load('weight.npy',allow_pickle=True)
w = weight[0]
x0 = np.zeros(2)
def Rosenfunc(x:np.array):
    return (1-x[0])**2 + w * (x[1] - x[0]**2)**2

epsilon = 0.1
def Rosengradient(x:np.array):
    return np.array([4*w*x[0]**3-4*w*x[0]*x[1]+2*x[0]-2,2*w*(x[1] - x[0] **2)])

def Rosenhessianmatrix(x:np.array):
    return np.array(
            [
                [12 * w * x[0]**2-4*w*x[1] +2,-4*w*x[0]],
                [-4*w*x[0],2*w]
            ]
        )
def Rosenapprox(x:np.array):
    global x0
    deltax = x - x0
    gradient = Rosengradient(x0)
    hessianmatrix = Rosenhessianmatrix(x0)
    return Rosenfunc(x0) + gradient.dot(deltax) + hessianmatrix.dot(deltax).dot(deltax)

def constrain(x:np.array):
    return sum((x - x0)**2)
trustregion = 0.01
maxtrustregion = 0.01
EPOCH = 256

from torch.utils.tensorboard import SummaryWriter
'''
|| x - x0 || <= trustregion
'''
writer = SummaryWriter('./log/differenttrustregionconstrain')
from tqdm import tqdm
for time in tqdm(range(EPOCH)):
    x = np.random.random(2)
    cons = NonlinearConstraint(constrain,-np.inf,trustregion)
    result = minimize(Rosenapprox,x0,method='SLSQP',constraints=cons)
    # print("result is ",result.x,result.message,result.fun)
    x = result.x
    rk = (Rosenfunc(x0) - Rosenfunc(x))/(Rosenfunc(x0) - Rosenapprox(x))
    print(rk)
    if rk < 0.25:
        trustregion = 0.25 * trustregion
        print("Small Trust Region")
    elif rk > 0.75:
        trustregion = min(2 * trustregion,maxtrustregion)
        print("Larger Trust Region")
    else:
        print("Same Trust Region")
    x0 = x
    writer.add_scalar('Rosenvalue',Rosenfunc(x0),time)
