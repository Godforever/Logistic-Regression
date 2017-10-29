import numpy as np
import math
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def Newton(dataMatIn, classLabels):
    maxCycles = 500000  # 设置最大迭代次数
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    w = np.ones((n, 1))
    k = 0
    hxhx_1 = lambda x:sigmoid(x)*(sigmoid(x)-1)#设置函数计算h(X)(h(X)-1)
    pr_w =  np.zeros((n, 1))
    while np.max(np.fabs(w-pr_w))>0.0001 and k < maxCycles:
        # 设置迭代终止条件为相邻参数差不超过0.0001或循环次数超过500000次
        h = sigmoid(dataMatrix * w)
        error = (labelMat - h)
        lw1 = 1/m*dataMatrix.transpose() * error
        t = np.array(h)
        A = np.identity(m)*hxhx_1(t)
        #计算黑塞尔矩阵
        Hmat = dataMatrix.transpose()*A*dataMatrix
        pr_w = w
        w = w - Hmat.I * lw1
        k += 1
    nx_loss_function = labelMat.transpose() * np.log(h) + (1-labelMat.transpose())*np.log(1-h)
    print(w)
    norm_2 = np.sqrt(np.array(w.transpose() * w)[0][0])  # 计算二范数
    print("二范数为", norm_2)
    print("似然函数为:", nx_loss_function)
    print("the iteration num is:", k)
    print()
    print(w)
    return w

def NewtonRegular(dataMatIn, classLabels):
    # 添加正则项的牛顿法算法
    lamda = 0.01  # 设置正则系数
    maxCycles = 500000  # 设置最大迭代次数
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    w = np.ones((n, 1))
    pr_loss_function = 0.0
    nx_loss_function = 1.0
    k = 0
    hxhx_1 = lambda x: sigmoid(x) * (sigmoid(x) - 1)  # 设置函数计算h(X)(h(X)-1)
    pr_w =  np.zeros((n, 1))
    while np.max(np.fabs(w-pr_w))>0.0001 and k < maxCycles:
        # 设置迭代终止条件为相邻参数差不超过0.0001或循环次数超过500000次
        norm_2 = np.sqrt(np.array(w.transpose()*w)[0][0])#计算二范数
        h = sigmoid(dataMatrix * w)
        error = (labelMat - h)
        lw1 = (dataMatrix.transpose() * error + lamda/norm_2*w)
        t = np.array(h)
        A = np.identity(m) * hxhx_1(t)
        # 计算添加正则项之后的黑塞尔矩阵
        Hmat = dataMatrix.transpose() * A * dataMatrix
        Hmat = Hmat + lamda / norm_2 * np.identity(n) - lamda / pow(norm_2, 1.5) * w.transpose() * w
        pr_w = w
        w = w - Hmat.I * lw1
        k += 1
    nx_loss_function = labelMat.transpose() * np.log(h) + (1-labelMat.transpose())*np.log(1-h)
    print(w)
    norm_2 = np.sqrt(np.array(w.transpose() * w)[0][0])  # 计算二范数
    print("二范数为", norm_2)
    print("似然函数为:", nx_loss_function)
    print("the iteration num is:", k)
    print()
    return w

