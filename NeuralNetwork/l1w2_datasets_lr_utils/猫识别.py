import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    return v



'''

- a training set of m_train images labeled as cat (y=1) or non-cat (y=0) 
- a test set of m_test images labeled as cat or non-cat 
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).

'''


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# 对数据的展示
# index = 25
# print(np.shape(train_set_x_orig))
# plt.imshow(train_set_x_orig[index])
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
# plt.show()

# 探索数据

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("训练样本数: m_train = " + str(m_train))
print ("测试样本数: m_test = " + str(m_test))
print ("照片维度: num_px = " + str(num_px))
print ("照片尺寸: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("训练集合特征: " + str(train_set_x_orig.shape))
print ("训练集合标签: " + str(train_set_y.shape))
print ("测试集合特征: " + str(test_set_x_orig.shape))
print ("测试集合标签: " + str(test_set_y.shape))

#对数据进行了平铺
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

print ("平铺后训练集合特征: " + str(train_set_x_flatten.shape))
print ("平铺后训练集合标签: " + str(train_set_y.shape))
print ("平铺后测试集合特征: " + str(test_set_x_flatten.shape))
print ("平铺后测试集合标签: " + str(test_set_y.shape))
print ("完整性检验: " + str(train_set_x_flatten[0:5,0]))

#照片像素归一化
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
'''
class LogisticRegression():
    def __init__(self):
        self.w = np.zeros(self.getShape(x)[0],1)
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
'''
def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X)+b)
    cost = -(1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    dw = (1.0/m)*np.dot(X,(A-Y).T)
    db = (1.0/m)*np.sum(A-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}

    return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):#np.where更好一点哦
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
index = 1
# plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
what = classes[int(d["Y_prediction_test"][0,index])].decode("utf-8")
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" +
       what +  "\" picture.")

# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()

# learning_rates = [0.01, 0.001, 0.0001]
# models = {}
# for i in learning_rates:
#     print ("learning rate is: " + str(i))
#     models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
#     print ('\n' + "-------------------------------------------------------" + '\n')
#
# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
#
# plt.ylabel('cost')
# plt.xlabel('iterations')
#
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()
#
#
# my_image = "cat.png"
my_image = "keji.png"
fname = "./" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)
plt.imshow(image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +
      "\" picture.")
plt.show()