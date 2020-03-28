import numpy as np
import math
import random
import matplotlib.pyplot as plt
'''
Linear and Logistic Regression:
    - Gradient Descent with decaying learning rate

Author: Me
'''

def sigmoid(z):
    return 1/(1+np.exp(-z))

def LRGradient(X, y, theta): #Linear Regression
    # X ~ [m x n+1]
    # y ~ [m x 1]
    m = len(y)
    h = X * theta
    grad = (1/m * X.transpose()*(h - y))
    err = h - y
    J = 1/(2*m) * (err).transpose() * (err)
    return grad, J
    
def LoGradient(X, y, theta): #Logistic Regression
    # X ~ [m x n+1]
    # y ~ [m x 1]
    m = len(y)
    h = sigmoid(X * theta) 
    grad = (1/m * X.transpose()*(h - y))
    J = 1/m * (y.transpose()*np.log(h)+(1-y).transpose()*np.log(1-h))
    return grad, J 

def GradientDescent(numiter, X, y, linear):
    m,n = np.shape(X)
    theta = np.random.rand(n,1)
    alpha = .5 #learning rate
    J_hist = np.ones(numiter)
    t_hist = np.ones(numiter)
    for t in range(numiter):
        if linear:
            grad, J = LRGradient(X, y, theta)
            theta2 = theta - alpha*grad 
            grad2, J2 = LRGradient(X, y, theta2) 
        else:
            grad, J = LoGradient(X, y, theta)
            theta2 = theta - alpha*grad
            grad2, J2 = LoGradient(X, y, theta2)
            
        while (J2 > J): #decrease alpha so it doesn't diverge
            alpha = .6*alpha
            theta2 = theta - alpha*grad 
            if linear:
                grad2, J2 = LRGradient(X, y, theta2)
            else:
                grad2, J2 = LoGradient(X, y, theta2)
        theta = theta2 - alpha*grad2
        
        if t % 30 == 0 and t != 0: # reset alpha
            alpha = .5
        J_hist[t] = J
        t_hist[t] = t
        
    return theta, J_hist, t_hist
        
def AddBias(X):
    m,n = np.shape(X)
    X0 = np.ones(m)
    X0 = X0[:,np.newaxis]
    Xn = np.append(X0, X, 1)
    return Xn

def AddFeature(X,x2):
    m,n = np.shape(X)
    Xn = np.append(X, x2, 1)
    return Xn

def MakeData(m):
    data = []
    for i in range(m):
        x = i
        y = 6*x**2
        row = [x, y]
        data.append(row)
    return data

def maintest():
    data = MakeData(30)
    MaTriX = np.matrix(data) 
    y = MaTriX.transpose()[-1].transpose()
    X = MaTriX.transpose()[:-1].transpose()

    X = AddFeature(X,np.square(X)) #append column vector x_2

    X = AddBias(X) #X ~ [m x 3]

    LinearRegression = True
    theta, J_hist, t_hist = GradientDescent(400, X, y, LinearRegression)
    print('Final Theta:',theta.transpose())
    print('correct Theta:',[0, 0, 6])
    plt.scatter(t_hist, J_hist)
    plt.show()

maintest()
