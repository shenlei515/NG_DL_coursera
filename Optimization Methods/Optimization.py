import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters,grads,learning_rate):
    for i in range(1,len(parameters)//2+1):
        parameters['W'+str(i)]=parameters['W'+str(i)]-learning_rate*grads['dW'+str(i)]
        parameters['b'+str(i)]=parameters['b'+str(i)]-learning_rate*grads['db'+str(i)]
    return parameters

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    #Step 2:partition
    num_partition=math.floor(X.shape[1]/mini_batch_size)
    for i in range(num_partition):
        batch_X=shuffled_X[:,i*mini_batch_size:(i+1)*mini_batch_size]
        batch_Y=shuffled_Y[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batches.append([batch_X,batch_Y])
    if X.shape[1]%mini_batch_size!=0:#注意条件
        batch_X=shuffled_X[:,num_partition*mini_batch_size+1:X.shape[1]]
        batch_Y=shuffled_Y[:,num_partition*mini_batch_size+1:Y.shape[1]]
        mini_batches.append([batch_X, batch_Y])
    return mini_batches

def initialize_velocity(parameters):
    V={}
    for i in range(1,len(parameters)//2+1):
        V['dW'+str(i)]=np.zeros(parameters['W'+str(i)].shape)
        V['db'+str(i)]=np.zeros(parameters['b'+str(i)].shape)

    return V

def update_parameters_with_momentum(parameters,v,dtheta,beta=0.9,learning_rate=0.01):
    for i in range(1,len(parameters)//2+1):
        v['dW'+str(i)]=beta*v['dW'+str(i)]+(1-beta)*dtheta['dW'+str(i)]
        v['db'+str(i)]=beta*v['db'+str(i)]+(1-beta)*dtheta['db'+str(i)]
        parameters['W'+str(i)]=parameters['W'+str(i)]-learning_rate*v['dW'+str(i)]
        parameters['b'+str(i)]=parameters['b'+str(i)]-learning_rate*v['db'+str(i)]
    return parameters,v

def initialize_adam(parameters):
    v={}
    s={}
    for i in range(1,len(parameters)//2+1):
        v["dW" + str(i)] = np.zeros(parameters["W" + str(i)].shape)
        v["db" + str(i)] = np.zeros(parameters["b" + str(i)].shape)
        s["dW" + str(i)] = np.zeros(parameters["W" + str(i)].shape)
        s["db" + str(i)] = np.zeros(parameters["b" + str(i)].shape)
    return v,s

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.007,beta1=0.9,beta2=0.99,epsilon=1e-7):
    for i in range(1, len(parameters) // 2 + 1):
        v['dW'+str(i)]=beta1*v['dW'+str(i)]+(1-beta1)*grads['dW'+str(i)]
        s['dW'+str(i)]=beta2*s['dW'+str(i)]+(1-beta2)*(grads['dW'+str(i)]**2)
        v['db'+str(i)]=beta1*v['db'+str(i)]+(1-beta1)*grads['db'+str(i)]
        s['db'+str(i)]=beta2*s['db'+str(i)]+(1-beta2)*(grads['db'+str(i)]**2)
        parameters['W'+str(i)]=parameters['W'+str(i)]-learning_rate*(v['dW'+str(i)]/(1-np.power(beta1,t)))/(np.sqrt(s['dW'+str(i)]/(1-np.power(beta2,t))+epsilon))
        parameters['b'+str(i)]=parameters['b'+str(i)]-learning_rate*(v['db'+str(i)]/(1-np.power(beta1,t)))/(np.sqrt(s['db'+str(i)]/(1-np.power(beta2,t))+epsilon))
    return parameters,v,s

train_X, train_Y = load_dataset()

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
plt.cla()
# # train 3-layer model
# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")
#
# # Predict
# predictions = predict(train_X, train_Y, parameters)
#
# # Plot decision boundary
# plt.title("Model with Gradient Descent optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5,2.5])
# axes.set_ylim([-1,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# # train 3-layer model
# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")
#
# # Predict
# predictions = predict(train_X, train_Y, parameters)
#
# # Plot decision boundary
# plt.title("Model with Adam optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5,2.5])
# axes.set_ylim([-1,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)