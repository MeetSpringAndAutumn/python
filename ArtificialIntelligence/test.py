import numpy as np
import operator
import pandas as pd


def KNN(testX, datasets, label, k):
    x = np.array([1, 2])
    y = np.random.randn(7, 2)
    d1 = np.sqrt(np.sum((x-y)**2, axis=1))
    d2 = np.sqrt((x[0]-y[:, 0])**2+(x[1]-y[:, 1])**2)
    size = y.shape[0]
    d3 = np.sqrt(np.sum((np.tile(x, (size, 1))-y)**2, axis=1))
    print(d1)
    print(d2)
    print(d3)
    sort_index = d1.argsort()
    label = [0, 1, 0, 1, 0, 0, 1]
    classlabel = {}
    k = 3
    for i in range(k):
        klabel = label[sort_index[i]]
        classlabel[klabel] = classlabel.get(klabel, 0)+1
        sortedclasslabel = sorted(
            classlabel.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedclasslabel[0][0])
    return sortedclasslabel[0][0]


np.loadtxt(path, delimiter, usecols, unpack)
data = np.loadtxt("D:\AI-ML\stock_data.csv", delimiter=',', dtype=str)
print(data[1:, 2:].astype(np.float32))
data = pd.read_excel("D:\AI-ML\stock_data.csv", sep=',')
