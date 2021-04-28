# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
np.random.seed(1) # set a seed so that the results are consistent

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
X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
#change the datasets
# def load_extra_datasets():
#     N = 200
#     noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
#     noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
#     blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
#     gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
#     no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
#
#     return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
#
# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
#
# datasets = {"noisy_circles": noisy_circles,
#             "noisy_moons": noisy_moons,
#             "blobs": blobs,
#             "gaussian_quantiles": gaussian_quantiles}
#
# dataset = "noisy_moons"
#
# X, Y = datasets[dataset]
# X, Y = X.T, Y.reshape(1, Y.shape[0])
#
# if dataset == "blobs":
#     Y = Y%2
#
# # Visualize the data
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);



shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size

clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

# Plot the decision boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")#准确率的计算方法是值为1，0的行向量和列向量相乘，分别得出为1的正确例子和0的正确例子，最后相加

#Defining the neural network structure
def model_struct(X,Y):
    n_x=shape_X[0]
    n_y=shape_Y[0]
    return n_x,n_y

def initialize(n_x,n_h,n_y):
    np.random.seed(2)
    w1=np.random.randn(n_h,n_x)
    b1=np.zeros((n_h,1))
    w2=np.random.randn(n_y,n_h)
    b2=np.zeros((n_y,1))
    return w1,b1,w2,b2

def forward_propaganda(X,parameters):
    w1,b1,w2,b2=parameters
    Z1=np.dot(w1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(w2,A1)+b2
    A2=sigmoid(Z2)
    return Z1,A1,Z2,A2

def compute_cost(Y,forwards):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(forwards[3]), Y) + np.multiply(np.log(1 - forwards[3]), (1 - Y))
    cost = np.squeeze(-(1.0 / m) * np.sum(logprobs))
    return cost

def back_propoganda(X,Y,forwards,parameters,learning_rate=1.2):
    dZ2=forwards[3]-Y
    dw2=1.0/m*(np.dot(dZ2,forwards[1].T))
    db2=1.0/m*(np.sum(dZ2,axis=1,keepdims=True))
    dZ1=(np.dot(parameters[2].T,dZ2))*(1-pow(forwards[1],2))
    dw1=1.0/m*np.dot(dZ1,X.T)
    db1=1.0/m*(np.sum(dZ1,axis=1,keepdims=True))
    w1=parameters[0]-learning_rate*dw1
    b1=parameters[1]-learning_rate*db1
    w2=parameters[2]-learning_rate*dw2
    b2=parameters[3]-learning_rate*db2
    return w1,b1,w2,b2

def nn_model(X,Y,n_h,num_iterations=10000):
    np.random.seed(3)
    n_x,n_y=model_struct(X,Y)
    parameters=initialize(n_x,n_h,n_y)
    for i in range(num_iterations):
        forwards=forward_propaganda(X,parameters)
        cost=compute_cost(Y,forwards)
        parameters=back_propoganda(X,Y,forwards,parameters,1.2)
        if i % 1000 == 0:
                    print ("Cost after iteration %i: %f" %(i, cost))
    return parameters

def predict(parameters,test):
    forwards=forward_propaganda(test,parameters)
    forwards=forwards[3][0]
    result=np.array(list(map(judge,forwards)))
    return result

def judge(x):
    if x<0.5:
        return 0
    else:
        return 1
#隐层神经元为4
# parameters=nn_model(X,Y,learning_rate=1.2,num_iterations=10000)
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()
#比较不同神经元数量的神经网络
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()