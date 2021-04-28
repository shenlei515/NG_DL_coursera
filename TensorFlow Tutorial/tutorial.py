import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

# # np.random.seed(1)
# #
# # y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
# # y = tf.constant(39, name='y')                    # Define y. Set to 39
# #
# # loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss
# #                                   # the loss variable will be initialized and ready to be computed
# # tf.print(loss)
#
# # a = tf.constant(2)
# # b = tf.constant(10)
# # c = tf.multiply(a,b)
# # print(c)
# #
# # x =tf.Variable(3,name='x')
# # print(2 * x)
# #
# # def linear_function():
# #     X = tf.constant(np.random.randn(3,1))
# #     W=tf.constant(np.random.randn(4,3),name="W")
# #     b=tf.constant(np.random.randn(4,1),name="b")
# #     Y=tf.matmul(W,X)+b
# #     return Y
# # tf.print(linear_function())
# #
# # def sigmoid(z):
# #     z=tf.constant(z,'float')
# #     s=tf.sigmoid(z)
# #     return s
# # print(sigmoid(0))#不能输入整数，貌似int32不属于参数的允许类型
# #
#
# def compute_cost(Z3,Y):
#     logits=Z3
#     labels=Y
#     cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,labels))
#     return cost
#
# # logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
# # cost = cost(logits, np.array([0,0,1,1]))#这里也不接受整数型的数据
# # print ("cost = " + str(cost))
# #
# # def one_hot_matrix(labels, C):
# #     labels=tf.constant(labels)
# #     depth=C
# #     one_hot=tf.one_hot(labels, depth,axis=0)#不能直接写个axis，再把axis直接作为参数写入，这样编译器会认为axis是第三个参数的值，而不是axis的值
# #     return one_hot
# # labels = np.array([1,2,3,0,2,1])
# # one_hot = one_hot_matrix(labels, C = 4)
# # print ("one_hot = " + str(one_hot))

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# #Flatten the training and test images
# X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1)
# X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1)
# # Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
# def initialize_parameters():
#     tf.random.set_seed(1)
#     init=tf.keras.initializers.GlorotUniform(seed=1)#xavier初始化器
#     W1 = init(shape=(25, 12288))
#     b1 = init(shape=(25, 1))
#     W2 = init(shape=(12, 25))
#     b2 = init(shape=(12, 1))
#     W3 = init(shape=(6, 12))
#     b3 = init(shape=(6, 1))
#     parameters={'W1':W1,'W2':W2,'W3':W3,'b1':b1,'b2':b2,'b3':b3}
#     return parameters
#
# def forward_propogation(X,parameters):
#     Z1=tf.matmul(parameters['W1'],X)+parameters['b1']
#     A1=tf.nn.relu(Z1)
#     Z2=tf.matmul(parameters['W2'],A1)+parameters['b2']
#     A2=tf.nn.relu(Z2)
#     Z3=tf.matmul(parameters['W3'],A2)+parameters['b3']
#     return Z3
#
# def model(X_train,Y_train,X_test,Y_test,learning_rate=0.0001,num_epochs=1500,minibatch_size=32,print_cost=True):
#     tf.random.set_seed(1)
#     seed=1
#     costs=[]
#     parameters=initialize_parameters()
#     W1=tf.Variable(parameters['W1'])
#     W2=tf.Variable(parameters['W2'])
#     W3=tf.Variable(parameters['W3'])
#     b1=tf.Variable(parameters['b1'])
#     b2=tf.Variable(parameters['b2'])
#     b3=tf.Variable(parameters['b3'])
#
#     for i in range(num_epochs):
#         e_cost=0
#         seed=seed+1
#         minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)#minibatch中有shuffle操作
#
#         for minibatch in minibatches:
#             (minibatch_X,minibatch_Y)=minibatch
#             X = tf.constant(minibatch_X,dtype=float)
#             Y = tf.constant(minibatch_Y,dtype=float)
#             cost = compute_cost(X,Y)
#             optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#             c=optimizer.minimize(cost,[W1,W2,W3,b1,b2,b3])#别用minimize这煞笔函数，参数乱七八糟
#             e_cost+=c/len(minibatches)
#         if(i%100==0 and print_cost==True):
#             print("Cost after epoch %i:%f" % (i,e_cost))
#         if print_cost==True and i%5==0:
#             costs.append(e_cost)
#     plt.plot(np.squeeze(costs))
#     plt.ylabel('cost')
#     plt.xlabel('iterations (per tens)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
#     predict_train=forward_propogation(X_train,parameters)
#     predict_test=forward_propogation(X_test,parameters)
#     correct_prediction_train = tf.equal(tf.argmax(predict_train), tf.argmax(Y_train))
#     accuracy_train = tf.reduce_mean(tf.cast(correct_prediction_train, "float"))
#     correct_prediction_test = tf.equal(tf.argmax(predict_test), tf.argmax(Y_test))
#     accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, "float"))
#     print("Train Accuracy:", accuracy_train)
#     print("Test Accuracy:", accuracy_test)
#
#     return parameters
#
# parameters = model(X_train, Y_train, X_test, Y_test)
Y_train=Y_train.T
Y_test=Y_test.T
print(X_train.shape,Y_train.shape)
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)).shuffle(1080).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(32)
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(64,64,3)),
    keras.layers.Dense(25,activation='relu',name='Dense_1'),
    keras.layers.Dense(12,activation='relu'),
    keras.layers.Dense(6,activation='softmax'),
])#输入形状对结果有影响？不是，是Flatten不负责把数据除225，导致初始值太大，loss一直不变
optimizer=tf.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),metrics=['categorical_accuracy'])
b_size = 32
max_epochs = 1500
print("Starting training ")
h = model.fit(train_ds, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("Training finished \n")
dense1=model.get_layer('Dense_1').variables#获取Dense_1层的参数
print(dense1)
prediction=model.predict(test_ds)
count=0
for i in range(Y_test_orig.shape[1]):
    if(Y_test_orig.T[i]==np.argmax(prediction[i])):
        count=count+1
    print(Y_test_orig.T[i],np.argmax(prediction[i]))
print("accuracy_inreal:%f" % (float(count)/float(Y_test_orig.shape[1])))
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
test_loss, test_acc =model.evaluate(test_ds)
print('\nTest accuracy:', test_acc)
#每次训练的结果相差过大，可能网络内有些参数随机初始化导致
plt.plot(np.squeeze(h.history))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()