import numpy as np

'''
X,y is training data and training observation value
,given a new data point x and return mu(x) and sigma(x)
'''
from math import exp 
from math import pi
from math import sqrt
def kernelfunction(x:np.ndarray,xstar:np.ndarray):
    matrix = []
    def _kernel(x_,xstar_,sigma=1):
        return 1/(sqrt(2*pi)*(sigma**2)) * exp((x_-xstar_)**2)
    for x_ in x:
        line = []
        for xstar_ in xstar:
            line.append(_kernel(x_,xstar_))
        matrix.append(line)
    
    return np.matrix(matrix)

def getmeanandsigma(X:np.ndarray,Y:np.ndarray,xstar:np.ndarray,gamma = 1,sigma=1):
    A = kernelfunction(X,X)+sigma ** 2*np.identity(len(X))
    A_1 = np.linalg.inv(A)
    # mu(x) = 0
    starxmatrix = kernelfunction(xstar,X)
    mux = starxmatrix.dot(A_1).dot(Y)
    # print(kernelfunction(xstar,xstar).shape)
    SIGMA = kernelfunction(xstar,xstar) - starxmatrix.dot(A_1).dot(kernelfunction(X,xstar))
    return mux, SIGMA
    
if __name__ == "__main__":
    N = 16
    X = np.random.random(N)
    Y = np.random.random(N)
    xstar = np.random.random(4)
    mux,SIGMA = getmeanandsigma(X,Y,xstar)
    print(mux.shape)
    print(SIGMA.shape)
    # x = np.array([1.0])
    # xstar = np.random.random(16)
    # matrix = kernelfunction(x,xstar)
    # print(matrix.shape)