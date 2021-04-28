import numpy as np
from testCases_2 import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

def forward_propagation(x, theta):
    f=theta*x
    return f
def back_propagation(x):
    dtheta=x
    return dtheta

def forward_propagation_n(X, Y, parameters):
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1./m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache

def backward_propagation_n(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def gradient_check(x):
    epsilon=1E-7
    gradapprox=(forward_propagation(x+epsilon)-forward_propagation(x-epsilon))/(2*epsilon)
    grad=back_propagation(x)
    accurate=np.linalg.norm(gradapprox-grad)/(np.linalg.norm(gradapprox)+np.linalg.norm(grad))
    if accurate < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")
    return accurate

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    parameters_values,_=dictionary_to_vector(parameters)
    print(parameters)
    grad=gradients_to_vector(gradients)
    num_parameters=parameters_values.shape[0]
    J_plus=np.zeros((num_parameters,1))
    J_minus=np.zeros((num_parameters,1))
    gradappox=np.zeros((num_parameters,1))
    for i in range(num_parameters):
        thetaplus=np.copy(parameters_values)
        thetaplus[i]=thetaplus[i]+epsilon
        J_plus[i],_=forward_propagation_n(X,Y,vector_to_dictionary(thetaplus))
        thetaminus=np.copy(parameters_values)
        thetaminus[i]=thetaminus[i]-epsilon
        J_minus[i],_=forward_propagation_n(X,Y,vector_to_dictionary(thetaminus))
        gradappox[i]=(J_plus[i]-J_minus[i])/(2*epsilon)
    print("grad="+str(grad))
    print("gradappox"+str(gradappox))
    difference=np.linalg.norm(grad-gradappox)/(np.linalg.norm(grad)+np.linalg.norm(gradappox))
    if difference > 1e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

X, Y, parameters = gradient_check_n_test_case()
cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)