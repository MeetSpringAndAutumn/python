import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# 读取数据
data = pd.read_csv('Mall_Customers.csv')

# 查看数据的前几行
# print(data.head())

# 数据可视化
sns.pairplot(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()

# 选取后三列数据作为训练数据
X = data.iloc[:, 2:5]

# 初始化列表存储inertia和轮廓系数得分
inertia = []
silhouette_scores = []

# 测试k值范围在2到11
k_range = range(2, 12)

for k in k_range:
    kmeans = KMeans(n_clusters=k,init='k-means++', random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# 绘制总距离平方和(inertia)的散点图
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Inertia vs. Number of Clusters')
plt.show()

# 绘制轮廓系数得分的条形图
plt.figure(figsize=(10, 6))
plt.bar(k_range, silhouette_scores)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

# 根据图选择最优K值，假设最优K值为7
optimal_k = 7

# 使用最优K值训练数据
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)

# 输出每个簇的中心点
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:\n", cluster_centers)

# 将中心点数据转换为DataFrame
cluster_centers_df = pd.DataFrame(cluster_centers, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

# 可视化簇的中心点
cluster_centers_df.plot(kind='bar', figsize=(12, 8))
plt.xticks(ticks=np.arange(optimal_k), labels=[f'Cluster {i+1}' for i in range(optimal_k)])
plt.title('Cluster Centers for Age, Annual Income, and Spending Score')
plt.xlabel('Cluster')
plt.ylabel('Value')
plt.show()

# 简单分析结果
for i, row in cluster_centers_df.iterrows():
    print(f"Cluster {i+1}:")
    print(f"  Age: {row['Age']}")
    print(f"  Annual Income (k$): {row['Annual Income (k$)']}")
    print(f"  Spending Score (1-100): {row['Spending Score (1-100)']}\n")
