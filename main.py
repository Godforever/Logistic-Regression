import numpy as np
import matplotlib.pyplot as plt
import method.GradientDescent as GD
import method.Newton as NT

def loadDataSet():

    dataMat = []
    labelMat = []
    fr = open('dataSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])#获取数据并添加常数项1，便于计算
        labelMat.append(int(lineArr[2]))#获取数据标签
    return dataMat, labelMat

def plotBestFit( dataMat, labelMat, w):
    #展示效果
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-4.0, 4.0, 0.1)
    y = (-w[0] - w[1] * x) / w[2]
    ax.plot(x, y)
    plt.title('Logistic Regression')
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


dataArr, labelMat = loadDataSet()

w = GD.GradientDescent(dataArr, labelMat)
plotBestFit(dataArr, labelMat,w.getA())

w_regular = GD.GradientDescentRegular(dataArr, labelMat)
plotBestFit(dataArr, labelMat,w_regular.getA())

w_N = NT.Newton(dataArr, labelMat)
plotBestFit(dataArr, labelMat,w_N.getA())

w_N_re = NT.NewtonRegular(dataArr, labelMat)
plotBestFit(dataArr, labelMat,w_N_re.getA())
