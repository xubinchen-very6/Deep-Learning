import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import matplotlib as mpl

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # 样本数量
    N = int(m/2) # 每个类别的样本量
    D = 2 # 维度数
    X = np.zeros((m,D)) # 初始化X
    Y = np.zeros((m,1), dtype='uint8') # 初始化Y
    a = 4 # 花儿的最大长度

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T
    return X, Y
x,y = load_planar_dataset()
class NeuralNetwork():
    def __init__(self, nh = 4, ny = 1):
        self.m  = self.getShape(x)[1]                   #400
        self.nx = self.getShape(x)[0]                   #2
        self.nh = nh                                    #4
        self.ny = ny                                    #1
        self.w1 = np.random.rand(self.nh,self.nx)  #4，2
        self.b1 = np.random.randn(self.nh,1)            #4，1
        self.w2 = np.random.rand(self.ny,self.nh)  #1，4
        self.b2 = np.random.randn(self.ny,1)            #1，1

    def getShape(self,x):
        return np.shape(x)[0],np.shape(x)[1]

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def tanh(self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

    def fit(self,x,y,eps = 0.1,epoch = 10000):
        for time in range(epoch):
            z1 = np.dot(self.w1,x)+self.b1                 #4，400
            a1 = self.tanh(z1)                             #4，400
            z2 = np.dot(self.w2,a1)+self.b2                #1，400
            a2 = self.sigmoid(z2)                          #1，400

            dz2 = a2 - y                                   #1，400
            dw2 = 1.0 / self.m * np.dot(dz2, a1.T)         #1，4
            db2 = 1.0 / self.m * np.sum(dz2, axis=1, keepdims=True)#1，1
            dz1 = np.dot(self.w2.T, dz2) * (1 - np.power(a1, 2))   #4，400
            dw1 = 1.0 / self.m * np.dot(dz1, x.T)                  #4，2
            db1 = 1.0 / self.m * np.sum(dz1, axis=1, keepdims=True)#4，1

            self.w1 -= eps*dw1
            self.w2 -= eps*dw2
            self.b1 -= eps*db1
            self.b2 -= eps*db2

            logprobs = np.multiply(np.log(a2), y) + np.multiply(np.log(1 - a2), (1 - y))
            cost = (-(1.0 / self.m) * np.sum(logprobs))

            if time%1000==0:
                print('epoch=%d cost=%.6f'%(time,cost))


        a2 = np.where(a2 >= 0.5,1,0)
        parameters = {'w1':self.w1,
                      'w2':self.w2,
                      'b1':self.b1,
                      'b2':self.b2}

        return parameters,a2

def score(a2,x,y):
    return 1-(np.sum(np.abs(a2-y))/np.shape(x)[1])

nn = NeuralNetwork()

# for i in [0.1,0.3,0.4,0.8,0.9,0.95,1.2,3]:
print('-'*30)
print()
# print('eps=',i)
parameters,a2 = nn.fit(x,y,eps=0.9)
scores = score(a2,x,y)
print('准确率为:',scores)


















