import numpy as np

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def GradientDescent(dataMatIn, classLabels):
    #梯度下降算法
    alpha = 0.1 #设置布长为0.1
    maxCycles = 500000 #设置最大迭代次数
    dataMatrix = np.mat(dataMatIn)#将数据转化为矩阵进行计算
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)#获取数据的维度
    w = np.ones((n, 1))
    k = 0
    pr_w =  np.zeros((n, 1))
    while np.max(np.fabs(w-pr_w))>0.0001 and k < maxCycles:
       # 设置迭代终止条件为相邻参数差不超过0.0001或循环次数超过500000次
        h = sigmoid(dataMatrix * w)
        error = (labelMat - h)
        pr_w = w
        w = w + alpha/m * dataMatrix.transpose() * error#更新参数w
        k += 1
    nx_loss_function = labelMat.transpose() * np.log(h) + (1-labelMat.transpose())*np.log(1-h)
    print(w)
    norm_2 = np.sqrt(np.array(w.transpose() * w)[0][0])  # 计算二范数
    print("二范数为", norm_2)
    print("似然函数为:", nx_loss_function)
    print("the iteration num is:", k)
    print()
    return w

def GradientDescentRegular(dataMatIn, classLabels):
    #添加正则项的梯度下降算法
    lamda = 0.1 #设置正则系数
    alpha = 0.1 #设置布长为0.001
    maxCycles = 500000 #设置最大迭代次数
    dataMatrix = np.mat(dataMatIn)  # 讲数据转化为矩阵进行计算
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)#获取数据维度
    w = np.ones((n, 1))
    k = 0
    pr_w =  np.zeros((n, 1))
    while np.max(np.fabs(w-pr_w))>0.0001 and k < maxCycles:
        # 设置迭代终止条件为相邻参数差不超过0.0001或循环次数超过500000次
        norm_2 = np.sqrt(np.array(w.transpose()*w)[0][0])#计算二范数
        h = sigmoid(dataMatrix * w)
        error = (labelMat - h)
        pr_w = w
        w = w + alpha/m * (dataMatrix.transpose() * error + lamda*1.0/norm_2*w)#更新参数w
        k += 1
    nx_loss_function = labelMat.transpose() * np.log(h) + (1-labelMat.transpose())*np.log(1-h)
    print(w)
    norm_2 = np.sqrt(np.array(w.transpose() * w)[0][0])  # 计算二范数
    print("二范数为", norm_2)
    print("似然函数为:", nx_loss_function)
    print("the iteration num is:", k)
    print()
    return w
