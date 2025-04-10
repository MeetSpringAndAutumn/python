import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取数据
data = np.loadtxt('data_multivar_cluster.txt',delimiter=',')

# 将数据可视化
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], s=50)
plt.title('Scatter Plot of Sample Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
# 使用K-means算法聚类数据
k = 4  # 簇的个数
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)

# 输出样本label及label的个数
labels = kmeans.labels_
unique_labels, label_counts = np.unique(labels, return_counts=True)
print("Sample labels:", unique_labels)
print("Number of samples in each cluster:", label_counts)

# 输出聚类后的中心点
centroids = kmeans.cluster_centers_
print("Cluster centroids:")
print(centroids)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='red', label='Centroids')
plt.title('Clustering Result with K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
