import numpy as np
a = np.array([[0,3,4],
              [2,6,4]])
a_bar = np.linalg.norm(a,axis=1,keepdims=True)

print(a)
print(a_bar)
print(a/a_bar)

#--------------------------softmax--------------------------
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
x_exp = np.exp(x)
x_exp_sum = np.sum(x_exp,axis=1,keepdims=True)

print(x_exp)
print(x_exp_sum)
print(x_exp/x_exp_sum)
#--------------------------L1-------------------------------
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
def l1(y,yhat):
    return np.abs(y-yhat)
result = l1(y,yhat)
print(result)