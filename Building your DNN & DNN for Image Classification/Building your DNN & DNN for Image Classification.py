import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters={}
    for i in range(1,len(layer_dims)):
        parameters['w'+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
        parameters['b'+str(i)]=np.zeros((layer_dims[i],1))
    return parameters

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['w' + str(l)] =  np.random.randn(layer_dims[l],layer_dims[l-1])/np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###

        assert(parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def linear_forward(A, W, b):
    Z=np.dot(W,A)+b
    cache=(A,W,b)
    return Z,cache

def linear_activation_forward(A_prev, W, b, activation):
    Z,linear_cache=linear_forward(A_prev,W,b)
    if activation=="sigmoid":
        A,activation_cache=sigmoid(Z)
    else:
        A,activation_cache=relu(Z)
    cache=(linear_cache,activation_cache)
    return A,cache

def forward_propagation(X,parameter):
    caches=[]
    A=X
    L=len(parameter)//2
    for i in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameter['w'+str(i)],parameter['b'+str(i)],"relu")
        caches.append(cache)
    AL,cache=linear_activation_forward(A,parameter['w'+str(L)],parameter['b'+str(L)],"sigmoid")
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y):
    m=Y.shape[1]
    cost=-1/m*(np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL)))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ,cache):
    A_prev,w,b=cache
    m=A_prev.shape[1]
    dw=1/m*np.dot(dZ,A_prev.T)
    db=1/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(w.T,dZ)
    return dA_prev,dw,db

def linear_activation_backward(dA,cache,activation):
    (linear_cache,activation_cache)=cache
    if activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
    else:
        dZ=relu_backward(dA,activation_cache)
    dA_prev,dw,db=linear_backward(dZ,linear_cache)
    return dA_prev,dw,db

def backward_propagantion(Y,caches,AL):
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    L=len(caches)
    grads={}
    grads["dA" + str(L-1)], grads["dw" + str(L)], grads["db" + str(L)]=linear_activation_backward(dAL,caches[L-1],"sigmoid")
    dA_prev=grads['dA' + str(L-1)]
    for i in reversed(range(L-1)):
        dA_prev,dw,db=linear_activation_backward(dA_prev,caches[i],"relu")
        grads['dw'+str(i+1)]=dw
        grads['db'+str(i+1)]=db
        grads['dA'+str(i)]=dA_prev
    return grads

def update_parameters(parameters,grads,learning_rate):
    for i in range(1,len(parameters)//2+1) :
        parameters['w'+str(i)]=parameters['w'+str(i)]-learning_rate*grads['dw'+str(i)]
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]
    return parameters

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
# Example of a picture
# index = 10
# plt.imshow(train_x_orig[index])
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# ### CONSTANTS DEFINING THE MODEL ####
# n_x = 12288     # num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)
# def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
#     np.random.seed(1)
#     grads = {}
#     costs = []  # to keep track of the cost
#     m = X.shape[1]  # number of examples
#     parameters=initialize_parameters(layers_dims)
#     for i in range(num_iterations):
#         AL,caches=forward_propagation(X,parameters)
#         cost=compute_cost(AL,Y)
#         grads=backward_propagantion(Y,caches,AL)
#         parameters=update_parameters(parameters,grads,learning_rate)
#         if print_cost and i % 100 == 0:
#             print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
#         if print_cost and i % 100 == 0:
#             costs.append(cost)
#     plt.plot(np.squeeze(costs))
#     plt.ylabel('cost')
#     plt.xlabel('iterations (per tens)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
#     return parameters
# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
#
#
# def predict(X, y, parameters):
#     """
#     This function is used to predict the results of a  L-layer neural network.
#
#     Arguments:
#     X -- data set of examples you would like to label
#     parameters -- parameters of the trained model
#
#     Returns:
#     p -- predictions for the given dataset X
#     """
#
#     m = X.shape[1]
#     n = len(parameters) // 2 # number of layers in the neural network
#     p = np.zeros((1,m))
#
#     # Forward propagation
#     probas, caches = forward_propagation(X, parameters)
#
#
#     # convert probas to 0/1 predictions
#     for i in range(0, probas.shape[1]):
#         if probas[0,i] > 0.5:
#             p[0,i] = 1
#         else:
#             p[0,i] = 0
#
#     print("Accuracy: "  + str(np.sum((p == y)/m)))
#
#     return p
#
# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)

layers_dims = [12288, 20, 7, 5, 1]

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    parameters=initialize_parameters_deep(layers_dims)
    for i in range(0,num_iterations):
        AL,caches=forward_propagation(X,parameters)
        cost=compute_cost(AL,Y)
        grads=backward_propagantion(Y,caches,AL)
        parameters=update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost=True)


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    # Forward propagation
    probas, caches = forward_propagation(X, parameters)


    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy: "  + str(np.sum((p == y)/m)))

    return p
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


def print_mislabeled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
        plt.show()
print_mislabeled_images(classes, test_x, test_y, pred_test)