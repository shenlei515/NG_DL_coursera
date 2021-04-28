import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

def zero_pad(X,pad):

    zero_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=((0,0),(0,0),(3,3),(0,0)))

    return zero_pad

def conv_single_step(a_slice_prev, W, b):
    res=np.sum(np.multiply(a_slice_prev,W))+b
    return res

def conv_forward(A_prev,W,b,hyperprameters):
    stride=hyperprameters['stride']
    pad=hyperprameters['pad']
    (m,n_h,n_w,n_c)=A_prev.shape
    (f,f,n_c,num)=W.shape
    a_prev_pad=zero_pad(A_prev,pad)
    vert_start=0
    vert_end=np.int((n_h-f+2*pad)/stride)+1
    horiz_start=0
    horiz_end=np.int((n_w-f+2*pad)/stride)+1
    res=np.zeros((m,vert_end,horiz_end,num))
    for k in range(m):
        a_prev_k=a_prev_pad[k,:,:,:]#循环的时候注意变量名如果取重复的，可能会引起数据改变，如果不是为了迭代，在多重循环中尽量不要使用重复的变量名自我赋值
        for i in range(vert_start,vert_end):
            for j in range(horiz_start,horiz_end):
                a_prev=a_prev_k[i*stride:i*stride+f,j*stride:j*stride+f,:]
                for n in range(num):
                    sin_res=conv_single_step(a_prev,W[:,:,:,n],b[0,0,0,n])
                    print(sin_res.shape)
                    res[k,i,j,n]=sin_res
    cache=(A_prev,W,b,hyperprameters)
    return res,cache

def pool_forward(A_prev,hyperparameters,mode="max"):
    (m,n_h,n_w,n_c)=A_prev.shape
    f=hyperparameters['f']
    stride=hyperparameters['stride']
    vert_start = 0
    vert_end = np.int((n_h - f) / stride) + 1
    horiz_start = 0
    horiz_end = np.int((n_w-f)/ stride) + 1
    A=np.zeros((m,vert_end,horiz_end,n_c))
    cache=(A_prev,hyperparameters)
    for k in range(m):
        a_prev=A_prev[k,:,:,:]
        for i in range(vert_start, vert_end):
            for j in range(horiz_start, horiz_end):
                a_prev_c= a_prev[i * stride:i * stride + f, j * stride:j * stride + f, :]
                for c in range(n_c):
                    a=a_prev_c[:,:,c]#这里一维和二维的冒号不能够省略，因为一二维取得不是单个值，而是一个方形范围，省略的话语法会认为你想取的是高上的c号数组
                    if mode=="max":
                        res=np.max(a)
                    else:
                        res=np.mean(a)
                    A[k,i,j,c]=res
    return A,cache