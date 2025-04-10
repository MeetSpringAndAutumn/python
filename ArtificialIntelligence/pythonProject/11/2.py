import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# 读取数据
data = np.loadtxt('data_multivar_cluster.txt',delimiter=',')

# 估计带宽（bandwidth）
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)

# 使用mean-shift算法聚类数据
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)

# 输出样本label及label的个数
labels = ms.labels_
unique_labels, label_counts = np.unique(labels, return_counts=True)
print("Sample labels:", unique_labels)
print("Number of samples in each cluster:", label_counts)

# 输出聚类后的中心点
centroids = ms.cluster_centers_
print("Cluster centroids:")
print(centroids)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='red', label='Centroids')
plt.title('Clustering Result with Mean Shift')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
