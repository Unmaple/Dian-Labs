import math
#from tqdm import tqdm
import numpy as np
import time

def dis(a,b):
    sum = 0
    for i in range(0,28):
        for j in range(0,28):
            sum += abs(a[i][j]-b[i][j])
    return sum

class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        # total = np.zeros(10)
        # Num = []
        # for i in range(0,10):
        #     Num.append(np.zeros((28,28)))
        # for i in range(0,60000):
        #     y1 = self.y[i]
        #     if total[y1] != 0 :
        #         Num[y1] = (Num[y1] * total[y1] + self.X[i]) / (total[y1] + 1)
        #     else :
        #         Num[y1] = self.X[i]
        #     total[y1] += 1









        result = np.ones(10000) * -1
        sum = 0
        for i in range(0,10000):
            neardi = 10 ** 8
            nearva = -1
            #for j in range(0,10):
                # sum = np.sum(np.abs((X[i]) - Num[j]))
                # if sum < neardi:
                #     nearva = j
                #     neardi = sum
            #start = time.time()
            for j in range(0,60000):
                # a = X[i]
                # b = self.X[j]
                # cha = a - b
                # chab = np.abs(cha)
                sum = np.sum(np.abs((X[i])-self.X[j]))


                # for k in range(0, 28):
                #     for l in range(0, 28):
                #         sum += abs(a[k][l] - b[k][l])
                if sum<neardi:
                    nearva = self.y[j]
                    neardi = sum
            # end = time.time()
            #
            # print(end - start)
            result[i] = nearva
            print(i,neardi,nearva)
        return result

        # raise NotImplementedError
        ...

        # End of todo
