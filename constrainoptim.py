from scipy.optimize import minimize
import numpy as np
e = 1e-10
func = lambda x: (x[0]-0.667)/(x[0] + x[1] + x[2] - 2)
cons = (
    {'type':'eq','fun':lambda x:x[0] * x[1] + x[2] - 1},
    {'type':'ineq','fun':lambda x: x[0] - e},
    {'type':'ineq','fun':lambda x: x[1] - e},
    {'type':'ineq','fun':lambda x: x[2] - e}
)
x0 = np.array((1.0,1.0,1.0))
result = minimize(func,x0,method='SLSQP',constraints=cons)
print(result.x,result.fun)