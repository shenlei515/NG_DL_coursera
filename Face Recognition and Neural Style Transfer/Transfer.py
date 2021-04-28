import os
import sys
import scipy.io
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

content_image = imageio.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)
imshow(content_image[0])

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

style_image = imageio.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])


generated_image = tf.convert_to_tensor([generated_image])
content_image = tf.convert_to_tensor([content_image])
style_image = tf.convert_to_tensor([style_image])


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.2),loss=lambda y_true,y_pred: y_pred)
model.fit(x=generated_image, epochs=200)


def compute_content_cost(a_C, a_G):
    shape = a_G.shape
    a_C = tf.reshape(a_C, [shape[1] * shape[2], shape[3]])
    a_G = tf.reshape(a_G, [shape[1] * shape[2], shape[3]])
    J_C = 1 / (4 * shape[1] * shape[2] * shape[3]) * (tf.reduce_sum(tf.square(a_C - a_G)))
    return J_C


def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def compute_layer_style_cost(a_S, a_G):
    shape = a_S.shape
    print(shape)
    a_S = tf.reshape(a_S, [shape[1] * shape[2], shape[3]])
    a_G = tf.reshape(a_G[0], [shape[1] * shape[2], shape[3]])
    GS = tf.matmul(tf.transpose(a_S), a_S)
    GG = tf.matmul(tf.transpose(a_G), a_G)
    J_S = 1 / (4 * np.square(shape[1] * shape[2] * shape[3])) * (tf.reduce_sum(tf.square(GS - GG)))
    return J_S


STYLE_LAYERS = [
    (4, 0.2),
    (21, 0.2),
    (38, 0.2),
    (71, 0.2),
    (104, 0.2)]


def compute_style_cost(a_S, a_G, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style0 = 0
    # if (np.array(a_S).shape == np.array(a_G).shape):
    #     print('equal')
    # else:
    #     print(0)
    for (layer_index, coeff) in STYLE_LAYERS:
        # Compute style_cost for the current layer
        # print(a_S[layer_index])
        # print(a_G[layer_index])
        J_style_layer = compute_layer_style_cost(a_S[layer_index], a_G[layer_index])

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style0 += coeff * J_style_layer

    return J_style0


def total_cost(J_content, J_style, alpha=10.0, beta=40.0):
    cost = alpha * J_content + beta * J_style
    return cost


class compute_content_cost1(tf.keras.layers.Layer):
    def __init__(self):
        super(compute_content_cost1, self).__init__()

    def __call__(self, inputs):
        return compute_content_cost(inputs[0], inputs[1])


class compute_style_cost1(tf.keras.layers.Layer):
    def __init__(self):
        super(compute_style_cost1, self).__init__()

    def __call__(self, inputs):
        return compute_style_cost(inputs[0], inputs[1], inputs[2])


class total_cost1(tf.keras.layers.Layer):
    def __init__(self):
        super(total_cost1,self).__init__()

    def __call__(self,inputs):
        return total_cost(inputs[0], inputs[1], inputs[2], inputs[3])


# 处理图片



# 计算a_C（内容在网络中输出的代表)
a_C0 = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(index=21).output)  # 可以通过这种方式取出部分模型
a_C = a_C0(content_image)
a_S = list(np.zeros((500,)))
# 计算a_S（风格图在网络中的输出）
for (i,weight) in STYLE_LAYERS:
    a_S0 = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(index=i).output)
    a_S[i]=a_S0(style_image)


class total_loss(tf.keras.layers.Layer):
    def __init__(self):
        super(total_loss, self).__init__()

    def __call__(self,gen):
        # #计算a_G(生成图（abstract）在网络中的输出）
        a_G = list(np.zeros((500,)))
        J_style=0
        for (i,weight) in STYLE_LAYERS:
            a_G0 = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(index=i).output)
            a_G[i] = a_G0(model.inputs)
            J_style+=weight*compute_layer_style_cost(a_S[i],a_G[i])
        # 计算总损失
        J_content = compute_content_cost1()([a_G[21], a_C])
        print(J_content)
        print(J_style)
        J_total = total_cost1()([J_content, J_style, 10.0, 40.0])
        return J_total


# 创建输入图形的占位层
# Input=tf.keras.Input((1,300,400,3))
# print(Input)
# generate=total_loss()(model.outputs)
# print("genetate"+str(generate))
# 建立输入生成图一输出的新模型
model1_generate = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
model1_generate.trainable=False


# save last generated image
save_image('generated_image.jpg', generated_image[0][0])