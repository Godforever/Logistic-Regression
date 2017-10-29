import numpy as np
import matplotlib.pyplot as plt
import method.GradientDescent as GD
import method.Newton as NT


def loadIrisDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        data = [1.0]
        data = data + [float(_s) for _s in lineArr[0:-1]]
        dataMat.append(data)#获取数据并添加常数项1，便于计算
        if(lineArr[-1].__eq__('Iris-setosa')):
            labelMat.append(1)
        else:
            labelMat.append(0)
    return dataMat, labelMat

#获取iris训练数据集
irisTrainDataArr, irisTrainLabel = loadIrisDataSet("iristrain.data")
#获取iris测试数据集
irisTestDataArr, irisTestLabel = loadIrisDataSet("iristest.data")

#采用梯度下降进行训练
w_Iris = GD.GradientDescentRegular(irisTrainDataArr, irisTrainLabel)

dataMat = np.mat(irisTestDataArr)
Z = np.array(dataMat*w_Iris)
predictLabel =[]
for z in Z:
    if(z>0):
        predictLabel.append(1)
    else:
        predictLabel.append(0)

ConfusionMat = np.zeros((2,2))
for i in range(predictLabel.__len__()):
    ConfusionMat[irisTestLabel[i]][predictLabel[i]] += 1

print("混淆矩阵为:")
print(ConfusionMat)



