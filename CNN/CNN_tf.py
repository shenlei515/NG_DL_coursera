import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
import tensorflow.python.keras.layers as layers
tf.keras.backend.clear_session()

np.random.seed(1)

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def compute_cost(Y, Z3):#y_ture和y_pred的顺序不能颠倒
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    ### END CODE HERE ###

    return cost


class loss_print(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.loss=[]
    def on_train_end(self, logs=None):
        plt.plot(np.squeeze(self.loss))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate = "+str(tf.keras.backend.get_value(self.model.optimizer.lr)))
        plt.show()
    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs.get('loss'))


class LearingRateSchedule(tf.keras.callbacks.Callback):#可以用Callback调整学习率，貌似也可以在优化器里调整
    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr,0.009*(1/(1+np.log(epoch)/3)))


tf.random.set_seed(1)                             # to keep results consistent (tensorflow seed)
seed = 3

inputs=tf.keras.Input(shape=(64,64,3))
h1=layers.Conv2D(8,kernel_size=[4,4],strides=[1,1],padding='SAME',kernel_initializer=tf.initializers.variance_scaling(seed = 0),activation='relu')(inputs)
h2=layers.MaxPool2D((8,8),strides=[8,8],padding='SAME')(h1)
h3=layers.Conv2D(16,kernel_size=[2,2],padding='SAME',strides=[1,1],kernel_initializer=tf.initializers.variance_scaling(seed = 0),activation='relu')(h2)
h4=layers.MaxPool2D((4,4),strides=[4,4],padding='SAME')(h3)
h5=layers.Flatten()(h4)
h6=layers.Dense(6)(h5)

model=tf.keras.Model(inputs=inputs,outputs=h6)
model.compile(optimizer='Adam',loss=compute_cost,metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=64,epochs=1000,callbacks=[loss_print(),LearingRateSchedule()],verbose=0)
test_scores=model.evaluate(X_test,Y_test)
print('test loss:', test_scores[0])
print('test acc:', test_scores[1])