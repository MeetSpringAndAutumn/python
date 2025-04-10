from sklearn import preprocessing
import numpy as np

# 读取数据
data = np.loadtxt('data_multivar.txt', delimiter=',')

# 均值移除（Mean Removal）
data_standard = preprocessing.scale(data)
# print(data_standard.mean(axis=0))
# 范围缩放（Min-Max Scaling）
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)

# 归一化（Normalization）
data_normalized = preprocessing.normalize(data, norm='l1')

# 二值化（Binarization）
threshold = np.mean(data_standard.mean(axis=0), axis=0)
# print(threshold)
data_binarizer = preprocessing.Binarizer(threshold=threshold)
data_binarized = data_binarizer.transform(data)

# 打印结果
print("均值移除后的数据：")
print(data_standard)
print("\n范围缩放后的数据：")
print(data_scaled)
print("\n归一化后的数据：")
print(data_normalized)
print("\n二值化后的数据：")
print(data_binarized)
