import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

# 读取图像文件
image = io.imread('cat.jpg')
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# 将图像从RGB转换为二维数组
image_2d = image.reshape((-1, 3))

# 使用K-means聚类算法
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(image_2d)
kmeans_labels = kmeans.predict(image_2d)
kmeans_centers = kmeans.cluster_centers_
# print(kmeans_centers.shape)
# print(type(kmeans_centers))
# print(kmeans_labels.shape)
# print(kmeans_labels[:10])
# print(kmeans_centers[kmeans_labels][:10])
# 替换像素点的值为其所属的聚类中心的值
kmeans_segmented_image = kmeans_centers[kmeans_labels].reshape(image.shape).astype(np.uint8)

# 显示K-means分割后的图像
plt.figure(figsize=(10, 10))
plt.imshow(kmeans_segmented_image)
plt.title('K-means Segmented Image')
plt.axis('off')
plt.show()

# 使用均值漂移聚类算法
bandwidth = estimate_bandwidth(image_2d, quantile=0.1, n_samples=100)
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift.fit(image_2d)
meanshift_labels = meanshift.predict(image_2d)
meanshift_centers = meanshift.cluster_centers_

# 替换像素点的值为其所属的聚类中心的值
meanshift_segmented_image = meanshift_centers[meanshift_labels].reshape(image.shape).astype(np.uint8)

# 显示均值漂移分割后的图像
plt.figure(figsize=(10, 10))
plt.imshow(meanshift_segmented_image)
plt.title('Mean Shift Segmented Image')
plt.axis('off')
plt.show()
# n_clusters_meanshift = len(np.unique(meanshift_labels))
# print(f"Mean Shift分割后的簇数: {n_clusters_meanshift}")