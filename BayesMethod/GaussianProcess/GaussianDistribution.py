from random import random
from math import sin

import matplotlib.pyplot as plt
import numpy as np
Noise = 100
Num_training = 4
Num_testing = 32
Extend = 128
def function(x):
    return sin(x) + random()/Noise

from math import exp
def distance(x1,x2,sigma=0.01):
    return exp(-1/(sigma**2)*((x1 - x2)**2))


def sample_data(num=Num_training):
    data = []
    label = []
    for _ in range(num):
        x = 2 * random() - 1
        data.append(x)
        label.append(function(x))

    return data, label
def K(data,data1):
    # data,label = sample_data()
    data = np.array(data)
    # label = np.array(label)

    Kmatrix = np.random.random((len(data),len(data1)))
    for i in range(len(data)):
        for j in range(len(data1)):
            Kmatrix[i][j] = distance(data[i],data1[j],sigma=1)
    return np.matrix(Kmatrix)
# def 
if __name__ == "__main__":
    traindata,trainlabel = sample_data(Num_training)

    testdata,testlabel = sample_data(Num_testing)
    matrix = K(traindata,traindata)
    testmatrix = K(testdata,traindata)
    plt.scatter(testdata,testlabel,c='r',linewidths=0.001)
    regressiondata = testmatrix.dot(matrix.I).dot(np.array(trainlabel))
    regressiondata = np.array(regressiondata)
    regressiondata = np.reshape(regressiondata,Num_testing)
    print('regression is',regressiondata)
    plt.scatter(testdata,regressiondata,c='b')
    plt.scatter(traindata,trainlabel,c='g',linewidths=0.001)
    plt.savefig("Bayes_Regression_4_zeromean.jpg")
    # print("regression is",regressiondata)

    



