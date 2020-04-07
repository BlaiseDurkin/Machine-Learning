import numpy as np
from numpy import linalg as la
import math
import random
import matplotlib.pyplot as plt
from operator import itemgetter
import time
'''
    Run-K-Means: runs K-Means with different K's
    K-Means: K-means with 4 random centroid initialize

Author: ~Yours truly~

    *NOTE* : K-Means is inconsistent at choosing best K
'''
def InitCentroids(K, X):
    m,n = np.shape(X)
    centroids = np.zeros((K, n))
    for k in range(K):
        centroids[k] = X[random.randint(0, m-2)] 
    return centroids
 
def K_means(K, X, max_iter, c_path):
    m,n = np.shape(X)
    class_i = np.zeros(m)
    attempts = 6
    JC = []
   
    for a in range(attempts):
        centroids = InitCentroids(K, X)
        for t in range(max_iter):
            for i in range(m):        # assign class 1:m
                distances = np.zeros(K)
                mind = float('inf')
                min_i = 0
                for j in range(K):
                    distances[j] = la.norm(X[i] - centroids[j])**2
                    if distances[j] < mind:
                        mind = distances[j]
                        max_i = j
                class_i[i] = max_i
            for k in range(K):        #   compute centroid 1:k
                try:
                    centroids[k] = np.mean(X[class_i == k], axis=0)
                except:
                    print('numpy has let you down...')
                error = la.norm(X[i] - centroids[j])**2
            J = 1/m * (np.sum(error))
            JC.append([J, centroids])
    
    sorted(JC, key=itemgetter(0))
    centroids = JC[0][1]
    J = JC[0][0]
    return centroids, J

def Run_K_means(X, max_iter, c_path):
    J_hist = []
    Ks = []
    for k in range(2,9):
        c, J = K_means(k, X, max_iter, c_path)
        J_hist.append(J)
        Ks.append(k)
        
        
    return Ks, J_hist
    

def make_data():
    n = 2
    m = 100
    data = []
    centers = [[10, 10], [20, 40], [30, 10]]
    for i in range(m):
        j = i%3
        example = [centers[j][0] + random.randrange(-8,8) , centers[j][1] + random.randrange(-8,8)]
        data.append(example)
    return data

def main_test():
    data = make_data()
    
    X = np.matrix(data)
    m,n = np.shape(X)
    c_path = False
    print('testing K-Means')
    Ks, J_hist = Run_K_means(X, 20, c_path)
   
    plt.scatter(Ks, J_hist)
    plt.show()
    print('Now with best K')
    #find K: min(J)
    minJ = 100000
    min_i = 0
    for i in range(len(J_hist)):
        if J_hist[i] < minJ:
            minJ = J_hist[i]
            min_i = i
    best_K = Ks[min_i]
    
    c, J = K_means(best_K, X, 20, c_path)

    print('best K:',best_K)
    
    plt.scatter(X.transpose()[0].tolist()[0], X.transpose()[1].tolist()[0])
    plt.scatter(c.transpose()[0].tolist()[:], c.transpose()[1].tolist()[:], color='r')
    plt.show()
    
main_test()



    
