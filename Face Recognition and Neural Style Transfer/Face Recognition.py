from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *


import cv2
np.set_printoptions(threshold=0.4)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())

# def triplet_loss(y_true, y_pred, alpha = 0.2):
#     """
#     Implementation of the triplet loss as defined by formula (3)
#
#     Arguments:
#     y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
#     y_pred -- python list containing three objects:
#             anchor -- the encodings for the anchor images, of shape (None, 128)
#             positive -- the encodings for the positive images, of shape (None, 128)
#             negative -- the encodings for the negative images, of shape (None, 128)
#
#     Returns:
#     loss -- real number, value of the loss
#     """
#
#     anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
#
#     ### START CODE HERE ### (≈ 4 lines)
#     # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
#     pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),axis=1)
#     # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
#     neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),axis=1)
#     # Step 3: subtract the two previous distances and add alpha.
#     basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
#     # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
#     loss = tf.reduce_sum(tf.maximum(basic_loss, 0.))
#     ### END CODE HERE ###
#
#     return loss

def triplet_loss(y_ture,y_pred,alpha=0.2):
    anchor,positive,nagative=y_pred[0],y_pred[1],y_pred[2]
    l_a_p=tf.convert_to_tensor((np.dot(np.array(anchor-positive),np.array(anchor-positive).T)))
    l_a_n=tf.convert_to_tensor((np.dot(np.array(anchor-nagative),np.array(anchor-nagative).T)))
    l=tf.reduce_sum(np.maximum(np.diagonal(l_a_p-l_a_n+alpha),0))
    print(np.diagonal(l_a_p-l_a_n+alpha))
    return l

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
# load_weights_from_FaceNet(FRmodel)

database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

def verify(image_path, identity, database, model):
    f_path=img_to_encoding(image_path,model)
    loss=np.linalg.norm(database[identity]-f_path)
    if loss<=0.7:
        door_open=True
    else:
        door_open=False

    return loss,door_open

verify("images/camera_0.jpg", "younes", database, FRmodel)

def who_is_it(image_path, database, model):
    f_path = img_to_encoding(image_path, model)
    min_dist=100
    for (name,db_enc) in database.item:
        if np.linalg.norm(db_enc-f_path)<min_dist:
            min_dist=np.linalg.norm(db_enc-f_path)
            identity=name
    if min_dist>0.7:
        print('Not In the database.')
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist,identity

# who_is_it("images/camera_0.jpg", database, FRmodel)