import numpy as np
from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
class LogisticRegression():
    def __init__(self):
        self.w = np.random.rand(self.getShape(x)[0],1)
        self.b = np.random.randn()

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def getShape(self,x):
        return np.shape(x)[0],np.shape(x)[1]

    def fit(self,x,y,eps=0.003,epoch=1000):
        for time in range(epoch):
            m = self.getShape(x)[1]
            z = np.dot(self.w.T,x) + self.b
            a = self.sigmoid(z)
            dz = a - y
            dw = (1/m)*np.dot(x,dz.T)
            db = (1/m)*np.sum(dz)
            self.w -= eps*dw
            self.b -= eps*db
        return self.w,self.b

    def predict(self,x,y):
        y_hat = self.sigmoid((np.dot(self.fit(x,y)[0].T,x)+self.fit(x,y)[1]))
        y_hat = np.where(y_hat>0.5,1,0)
        return y_hat

    def score(self,x,y):
        # print(str(1-(np.sum(np.abs(self.predict(x,y)-y))/self.getShape(x)[1])*100)+'%')
        return (str((1-(np.sum(np.abs(self.predict(x, y) - y)) / self.getShape(x)[1]))*100)+('%'))

data = load_iris()
x = data.data[:100,0:2]
y = data.target[:100]
x = x.T

lr = LogisticRegression()
w,b = lr.fit(x,y)
y_hat = lr.predict(x,y)
print(y_hat)
print(y)
print(lr.score(x,y))




