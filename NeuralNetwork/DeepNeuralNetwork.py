import numpy as np
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

def initParameters(layer_dims):
    L = len(layer_dims)
    parameters = {}
    for l in range(1,L):
        parameters['w'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters['b'+str(l)] = np.random.rand(layer_dims[l],1)
    return parameters
def sigmoid(z):
    cache = z
    a = 1/(1+np.exp(-z))
    return a,cache
def sigmoidDerivation(da,cache):
    z = cache
    s = 1/(1+np.exp(-z))
    dz = da*s*(1-s)
    return dz
def relu(z):
    cache = z
    a = np.maximum(0,z)
    return a,cache
def reluDerivation(da,cache):
    z = cache
    # print(np.shape(z))
    dz = np.array(da,copy = True)
    # print(np.shape(dz))
    dz[z <= 0] = 0
    return dz
def linear_forward(x,w,b):
    z = np.dot(w,x)+b
    cache = (x,w,b)
    return z,cache
def linear_activation_forward(a_pre,w,b,activation):

    if activation == 'relu':
        z,linear_cache = linear_forward(a_pre,w,b)
        a,activation_cache = relu(z)
    elif activation == 'sigmoid':
        z,linear_cache = linear_forward(a_pre,w,b)
        a,activation_cache = sigmoid(z)
    cache = (linear_cache,activation_cache)
    return a,cache
def l_model_forward(x,parameters):
    caches = []
    L = len(parameters)//2
    a = x
    for l in range(1,L):
        a_pre = a
        a,cache = linear_activation_forward(a_pre,parameters['w'+str(l)],parameters['b'+str(l)],'relu')
        caches.append(cache)
    a,cache = linear_activation_forward(a,parameters['w'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    return a,caches
def linear_backward(dz,cache):
    a_pre,w,b = cache
    m = np.shape(a_pre)[1]
    dw = np.dot(dz,a_pre.T)/m
    db = np.sum(dz,axis=1,keepdims=True)/m
    da_pre = np.dot(w.T,dz)
    return da_pre,dw,db
def linear_activation_backward(da,caches,activation):
    linear_cache,activation_cache = caches
    if activation == 'sigmoid':
        dz = sigmoidDerivation(da,activation_cache)
        da_pre,dw,db = linear_backward(dz,linear_cache)
    elif activation == 'relu':
        dz = reluDerivation(da,activation_cache)
        da_pre,dw,db = linear_backward(dz,linear_cache)
    return da_pre,dw,db
def l_model_backward(al,y,caches):
    grads = {}
    L = len(caches)
    y = y.reshape(np.shape(al))

    dal = -(np.divide(y,al)-np.divide(1-y,1-al))
    current_cache = caches[L-1]
    grads['da'+str(L)],grads['dw'+str(L)],grads['db'+str(L)] = linear_activation_backward(dal,current_cache,'sigmoid')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['da' + str(l+1)], grads['dw' + str(l+1)], grads['db' + str(l+1)] = linear_activation_backward(grads['da'+str(l+2)],current_cache,'relu')
    return grads

def update(parameters,grads,eps=0.02):
    L = len(parameters)//2
    for l in range(L):
        parameters['w'+str(l+1)] -= eps * grads['dw'+str(l+1)]
        parameters['b'+str(l+1)] -= eps * grads['db'+str(l+1)]
    return parameters

parameters = initParameters([2,4,5,1])
for i in range(15000):
    al,caches = l_model_forward(x,parameters)
    grads = l_model_backward(al,y,caches)
    parameters = update(parameters,grads,eps=0.52)
y_hat = np.where(al>0.5,1,0)
print(1-(np.sum(np.abs(y_hat-y))/400))














