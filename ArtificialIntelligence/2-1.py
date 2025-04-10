import operator
import numpy as np
import os
import sys
os.chdir(sys.path[0])


def knn(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 行数
    # 计算欧式距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - \
        dataSet  # 扩展dataSet行，分别相减，形成（x1-x2）矩阵
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  # 从小到大排序，获得索引值(下标）
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return int(sortedClassCount[0][0])


# 读取数据
data = np.loadtxt(
    "D:\\Desktop\\python\\ArtificialIntelligence\\datingTestSet2.txt", delimiter="	")

# 分割数据
label = data[:, -1]
data = data[:, :3]


# 训练模型
# 这里没有使用模型训练，而是直接将训练数据传入KNN函数

# 选择一个样本进行预测
sample = data[0]

# 预测结果
prediction = knn(sample, data, label, k=3)

# 输出预测结果
print(prediction)
